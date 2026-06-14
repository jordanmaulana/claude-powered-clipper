import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np

from lib.facetrack import (
    MOTION_FLOOR,
    MOUTH_PATCH,
    SPEAKER_DWELL,
    SPEAKER_MARGIN,
    LINK_MAX_JUMP,
    STICKINESS,
    _best_by_area,
    _extract_mouth_patch,
    _link_detections,
    _mouth_motion,
    _select_speaker,
    _smooth,
    _step,
)


def spk(id, cx, motion, *, miss=0, area=0.2, conf=0.9):
    """A tracklet for the speaker selector."""
    return {"id": id, "cx": float(cx), "smooth_motion": float(motion),
            "miss": miss, "area": float(area), "conf": float(conf)}


def fresh_state():
    return {"active_id": None, "challenger_id": None, "challenge_count": 0}


def trk(cx, id):
    return {"cx": float(cx), "id": id}


def det(cx):
    return {"cx": float(cx)}


def cand(cx, *, area, conf=0.9, id=0):
    """A candidate face for _best_by_area (cx full-res, normalized area)."""
    return {"cx": float(cx), "area": float(area), "conf": float(conf), "id": id}


def face_row(x, y, w, h, *, conf=0.9, rmouth=None, lmouth=None):
    """Build a YuNet-style 15-float face row (small-frame coords)."""
    cx = x + w / 2
    mouth_y = y + h * 0.75
    rmx, rmy = rmouth if rmouth else (cx - w * 0.2, mouth_y)
    lmx, lmy = lmouth if lmouth else (cx + w * 0.2, mouth_y)
    return np.array([
        x, y, w, h,
        cx - w * 0.2, y + h * 0.4,   # right eye
        cx + w * 0.2, y + h * 0.4,   # left eye
        cx, y + h * 0.55,            # nose
        rmx, rmy,                    # right mouth corner
        lmx, lmy,                    # left mouth corner
        conf,
    ], dtype=np.float32)


class TestMouthMotion:
    def test_identical_patches_zero(self):
        patch = np.full((MOUTH_PATCH, MOUTH_PATCH), 120, dtype=np.uint8)
        assert _mouth_motion(patch, patch.copy()) == 0.0

    def test_none_patch_zero(self):
        patch = np.zeros((MOUTH_PATCH, MOUTH_PATCH), dtype=np.uint8)
        assert _mouth_motion(None, patch) == 0.0
        assert _mouth_motion(patch, None) == 0.0

    def test_changed_patch_positive_normalized(self):
        a = np.zeros((MOUTH_PATCH, MOUTH_PATCH), dtype=np.uint8)
        b = np.full((MOUTH_PATCH, MOUTH_PATCH), 255, dtype=np.uint8)
        assert _mouth_motion(a, b) == 1.0  # full swing -> 1.0
        half = np.full((MOUTH_PATCH, MOUTH_PATCH), 128, dtype=np.uint8)
        m = _mouth_motion(a, half)
        assert 0.0 < m < 1.0


class TestBestByArea:
    def test_no_prev_picks_largest(self):
        faces = [cand(100, area=0.10, id=1), cand(500, area=0.30, id=2)]
        assert _best_by_area(faces, None, 1000)["id"] == 2

    def test_stickiness_prefers_near_previous(self):
        # smaller face right at prev_cx beats a slightly bigger far face
        faces = [cand(500, area=0.20, id=1), cand(100, area=0.22, id=2)]
        # prev at 500: face 1 dist 0, face 2 dist 0.4 -> penalty STICKINESS*0.4
        chosen = _best_by_area(faces, 500, 1000)
        assert STICKINESS * 0.4 > (0.22 - 0.20)  # sanity: penalty dominates
        assert chosen["id"] == 1

    def test_empty_returns_none(self):
        assert _best_by_area([], None, 1000) is None


class TestLinkDetections:
    def test_one_to_one_match(self):
        tracklets = [trk(100, 1), trk(500, 2)]
        dets = [det(110), det(490)]
        matched = _link_detections(tracklets, dets, 1000)
        assert matched == {0: 0, 1: 1}  # det0->trk0(id1), det1->trk1(id2)

    def test_beyond_gate_unmatched(self):
        tracklets = [trk(100, 1)]
        # gate = 0.18*1000 = 180; det at 400 is 300 away -> no match
        dets = [det(400)]
        matched = _link_detections(tracklets, dets, 1000)
        assert matched == {}

    def test_greedy_no_double_assign(self):
        # two dets both nearer trk1 than trk2; greedy assigns nearest globally first
        tracklets = [trk(100, 1), trk(120, 2)]
        dets = [det(105), det(125)]
        matched = _link_detections(tracklets, dets, 1000)
        assert sorted(matched.keys()) == [0, 1]
        assert sorted(matched.values()) == [0, 1]  # bijection, no tracklet reused

    def test_gate_boundary(self):
        tracklets = [trk(100, 1)]
        dets = [det(100 + LINK_MAX_JUMP * 1000 + 1)]  # just beyond gate
        assert _link_detections(tracklets, dets, 1000) == {}


class TestSelectSpeaker:
    M = 0.05  # motion well above MOTION_FLOOR

    def test_no_live_returns_none(self):
        state = fresh_state()
        assert _select_speaker([spk(1, 100, self.M, miss=2)], state, 1000) is None

    def test_single_live_returns_its_cx(self):
        state = fresh_state()
        assert _select_speaker([spk(1, 300, self.M)], state, 1000) == 300.0
        assert state["active_id"] == 1

    def test_ambiguous_motion_uses_area_fallback(self):
        # both below MOTION_FLOOR -> pick largest area (id 2), not motion leader
        state = fresh_state()
        live = [spk(1, 100, 0.001, area=0.10), spk(2, 500, 0.002, area=0.30)]
        assert _select_speaker(live, state, 1000) == 500.0
        assert state["active_id"] == 2

    def test_holds_active_until_dwell_then_switches(self):
        state = fresh_state()
        a = spk(1, 100, self.M)              # active, steady talker
        b = spk(2, 900, self.M * 2)          # challenger, clearly louder
        # first sample establishes active=1 (a leads initially)
        a_lead = [spk(1, 100, self.M * 2), spk(2, 900, self.M)]
        assert _select_speaker(a_lead, state, 1000) == 100.0
        assert state["active_id"] == 1
        # now b leads by > MARGIN; must hold A for DWELL-1 samples
        for i in range(SPEAKER_DWELL - 1):
            assert _select_speaker([a, b], state, 1000) == 100.0, f"sample {i}"
        # DWELL-th qualifying sample: switch to B
        assert _select_speaker([a, b], state, 1000) == 900.0
        assert state["active_id"] == 2

    def test_challenge_resets_when_active_regains(self):
        state = {"active_id": 1, "challenger_id": None, "challenge_count": 0}
        a = spk(1, 100, self.M)
        b = spk(2, 900, self.M * 2)
        for _ in range(SPEAKER_DWELL - 1):   # b leads, but not long enough
            _select_speaker([a, b], state, 1000)
        assert state["challenge_count"] == SPEAKER_DWELL - 1
        # a roars back -> no switch, challenge resets
        assert _select_speaker([spk(1, 100, self.M * 3), b], state, 1000) == 100.0
        assert state["active_id"] == 1
        assert state["challenge_count"] == 0

    def test_active_disappears_adopts_leader(self):
        state = {"active_id": 1, "challenger_id": None, "challenge_count": 0}
        # tracklet 1 not present (left frame); 2 and 3 live, 3 louder
        live = [spk(2, 400, self.M), spk(3, 800, self.M * 2)]
        assert _select_speaker(live, state, 1000) == 800.0
        assert state["active_id"] == 3

    def test_active_speaker_disabled_uses_v1_scorer(self):
        # loud challenger present, but active_speaker=False must ignore lip motion
        state = {"active_id": 1, "challenger_id": None, "challenge_count": 0}
        live = [spk(1, 100, self.M, area=0.10), spk(2, 900, self.M * 5, area=0.40)]
        for _ in range(SPEAKER_DWELL + 3):
            cx = _select_speaker(live, state, 1000, active_speaker=False)
        assert cx == 100.0  # v1 stickiness holds subject 1, ignores loud mouth
        assert state["active_id"] == 1

    def test_margin_required_no_switch_when_close(self):
        state = {"active_id": 1, "challenger_id": None, "challenge_count": 0}
        a = spk(1, 100, self.M)
        b = spk(2, 900, self.M * 1.1)  # leads but below SPEAKER_MARGIN (1.30)
        for _ in range(SPEAKER_DWELL + 3):
            assert _select_speaker([a, b], state, 1000) == 100.0
        assert state["active_id"] == 1


class TestStepBackCompat:
    """One face per sample must reproduce the v1 single-subject center exactly."""

    def _frame(self, h=360, w=640):
        col = np.linspace(0, 255, w, dtype=np.uint8)
        gray = np.tile(col, (h, 1))
        return np.repeat(gray[:, :, None], 3, axis=2)

    def test_single_face_stream_matches_v1(self):
        small = self._frame()
        scale = 0.5            # 640 detect width / 1280 full width
        width = 1280.0
        det_area = 640 * 360
        # a face that drifts a little across samples (still one subject)
        xs = [280, 300, 290, 310, 305]
        tracklets, state, next_id = [], fresh_state(), 0
        for x in xs:
            f = face_row(x, 120, 80, 100)
            center, next_id = _step(small, [f], scale, det_area, width,
                                    tracklets, state, next_id)
            expected = (x + 80 / 2) / scale  # v1: (x + w/2)/scale of the lone face
            assert center == expected

    def test_no_faces_returns_none_center(self):
        small = self._frame()
        tracklets, state, next_id = [], fresh_state(), 0
        center, next_id = _step(small, [], 0.5, 640 * 360, 1280.0,
                                tracklets, state, next_id)
        assert center is None


class TestSmoothSnapFlag:
    def test_no_snap_lags_on_subthreshold_jump(self):
        width = 1000
        times = [0, 0.2, 0.4, 0.6]
        centers = [100, 100, 200, 200]  # jump 100 = 0.1w < SNAP_JUMP (0.25w)
        out = _smooth(times, centers, width)
        assert out[2] < 150  # EMA lags, no snap

    def test_snap_flag_forces_jump(self):
        width = 1000
        times = [0, 0.2, 0.4, 0.6]
        centers = [100, 100, 200, 200]
        out = _smooth(times, centers, width, snap_flags=[False, False, True, False])
        assert out[2] == 200  # forced snap to raw center

    def test_snap_flags_default_none_unchanged(self):
        width = 1000
        times = [0, 0.2, 0.4]
        centers = [100, 100, 130]
        assert _smooth(times, centers, width) == _smooth(times, centers, width, None)


class TestExtractMouthPatch:
    def _frame(self, h=360, w=640):
        # deterministic gradient so a real ROI is non-degenerate
        col = np.linspace(0, 255, w, dtype=np.uint8)
        gray = np.tile(col, (h, 1))
        return np.repeat(gray[:, :, None], 3, axis=2)

    def test_returns_fixed_size_uint8(self):
        small = self._frame()
        f = face_row(280, 120, 80, 100)
        patch = _extract_mouth_patch(small, f)
        assert patch is not None
        assert patch.shape == (MOUTH_PATCH, MOUTH_PATCH)
        assert patch.dtype == np.uint8

    def test_profile_face_returns_none(self):
        small = self._frame()
        # mouth corners collapsed (profile) -> mouth_dist < 2px
        f = face_row(280, 120, 80, 100, rmouth=(320.0, 195.0), lmouth=(320.5, 195.0))
        assert _extract_mouth_patch(small, f) is None

    def test_edge_corners_never_raise(self):
        small = self._frame()
        f = face_row(-30, -20, 60, 60)  # bbox off the top-left edge
        # must not raise; returns a patch or None
        _extract_mouth_patch(small, f)
