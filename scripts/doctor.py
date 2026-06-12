"""Environment check: ffmpeg, whisper backend, YuNet model download."""

import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import ffmpeg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK ' if ok else 'FAIL'}  {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> None:
    print("aiclipper doctor\n")
    ok = True

    ok &= check("ffmpeg", ffmpeg.FFMPEG is not None, str(ffmpeg.FFMPEG))
    ok &= check("ffprobe", ffmpeg.FFPROBE is not None)
    if ffmpeg.FFMPEG:
        filters = subprocess.run(
            [ffmpeg.FFMPEG, "-hide_banner", "-filters"], capture_output=True, text=True
        ).stdout
        ok &= check("libass (caption burning)", " ass " in filters)
        ok &= check("sendcmd (dynamic crop)", "sendcmd" in filters)

    try:
        import mlx_whisper  # noqa: F401
        ok &= check("whisper backend", True, "mlx-whisper (Apple Silicon)")
        print("        note: first transcription downloads ~1.6 GB model weights from HuggingFace")
    except ImportError:
        try:
            import faster_whisper  # noqa: F401
            ok &= check("whisper backend", True, "faster-whisper")
        except ImportError:
            ok &= check("whisper backend", False, "run: uv sync")

    if not MODEL_PATH.exists():
        print(f"  ...   downloading YuNet model (~345 KB) -> {MODEL_PATH}")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    try:
        import cv2
        det = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (320, 320))
        ok &= check("YuNet face detector", det is not None, MODEL_PATH.name)
    except Exception as e:  # noqa: BLE001
        ok &= check("YuNet face detector", False, str(e))

    print()
    if not ok:
        sys.exit("doctor: FAILED — fix the items above")
    print("doctor: all checks passed")


if __name__ == "__main__":
    main()
