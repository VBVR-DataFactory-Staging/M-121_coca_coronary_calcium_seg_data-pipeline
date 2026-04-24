"""Video + overlay primitives for the KneeXray pipeline (Linux-safe ffmpeg)."""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


KNEE_COLORS = [(0, 200, 0), (0, 128, 255)]  # BGR — left=green, right=orange


def loop_frames(image: np.ndarray, n: int = 6) -> List[np.ndarray]:
    return [image.copy() for _ in range(n)]


def draw_bbox(
    image: np.ndarray,
    box_xyxy: Sequence[float],
    color: Tuple[int, int, int],
    label: str = "",
    thickness: int = 3,
) -> np.ndarray:
    out = image.copy()
    x1, y1, x2, y2 = (int(round(v)) for v in box_xyxy)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        tk = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, tk)
        ty = max(th + 4, y1 - 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(out, label, (x1 + 3, ty - 2), font, scale,
                    (255, 255, 255), tk, cv2.LINE_AA)
    return out


def overlay_all(
    image: np.ndarray,
    bboxes: Sequence[Sequence[float]],
    grades: Sequence[int],
    side_labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Draw every bbox + KL label."""
    out = image.copy()
    for i, (b, g) in enumerate(zip(bboxes, grades)):
        color = KNEE_COLORS[i % len(KNEE_COLORS)]
        side = side_labels[i] if side_labels and i < len(side_labels) else f"Knee {i+1}"
        text = f"{side}  KL={g}"
        out = draw_bbox(out, b, color, label=text)
    return out


def make_video(frames: List[np.ndarray], out_path: Path, fps: int = 6) -> None:
    if not frames:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    w2, h2 = w - (w % 2), h - (h % 2)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-vf", f"scale={w2}:{h2}",
        str(out_path),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()


def upscale(img: np.ndarray, target: int = 640) -> np.ndarray:
    """Upscale small (256x320) X-rays so overlays + text are legible."""
    h, w = img.shape[:2]
    scale = target / min(h, w)
    if scale <= 1.0:
        return img
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_CUBIC)


def to_uint8_bgr(image: np.ndarray) -> np.ndarray:
    """KneeXray H5 stores float64 in [0, 255]; bring to uint8 BGR."""
    arr = image
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[-1] == 3:
        # H5 layout is RGB-ish from PIL save; treat channels as already
        # interchangeable since the X-ray is grayscale-replicated.
        return arr.copy()
    return arr.copy()
