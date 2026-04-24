"""TaskPipeline for M-121 KneeXray (knee localization + KL grading).

Lives at the M-121_coca slot — bootstrap uploads /tmp/out/ to
``s3://vbvr-final-data/questions/M-121_coca_coronary_calcium_seg_data-pipeline/``.
The Phase-2 ``_example_output`` path is intentionally not used here; we go
through the standard Phase-1 generate.py → /tmp/out flow so there is no
output-key mismatch.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from core.pipeline import BasePipeline, TaskSample
from src.download.downloader import create_downloader, KL_GRADE_NAMES
from .config import TaskConfig
from .transforms import (
    KNEE_COLORS,
    draw_bbox,
    loop_frames,
    make_video,
    overlay_all,
    to_uint8_bgr,
    upscale,
)


def _side_label(box_xyxy, image_w):
    """Heuristic: knee with smaller x1 = LEFT (patient's left = image right
    in radiology convention, but for natural language we just call them
    'Left of image' / 'Right of image' to avoid ambiguity)."""
    x1 = box_xyxy[0]
    return "Left" if x1 + (box_xyxy[2] - x1) / 2 < image_w / 2 else "Right"


class TaskPipeline(BasePipeline):
    def __init__(self, config: Optional[TaskConfig] = None):
        super().__init__(config or TaskConfig())
        self.downloader = create_downloader(self.config)

    def download(self) -> Iterator[dict]:
        yield from self.downloader.iter_samples(limit=self.config.num_samples)

    # ---------------------------------------------------------------- helpers

    def _build_frames(self, raw: dict):
        img = to_uint8_bgr(raw["image"])
        h0, w0 = img.shape[:2]
        img = upscale(img, target=640)
        h1, w1 = img.shape[:2]
        sx, sy = w1 / w0, h1 / h0

        # rescale boxes
        boxes = []
        for x1, y1, x2, y2 in raw["bboxes_xyxy"]:
            boxes.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
        grades = list(raw["kl_grades"])

        # left/right labels
        # Sort boxes left-to-right so colour assignment is stable
        order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
        boxes = [boxes[i] for i in order]
        grades = [grades[i] for i in order]
        sides = []
        for b in boxes:
            sides.append(_side_label(b, w1))

        fps = self.config.fps

        # First segment: 1 sec of plain X-ray
        first_frames = loop_frames(img, n=fps)

        # Ground-truth animation:
        #   stage 1: plain image (1s)
        #   stage 2: bbox-only reveal, knees added one at a time (fps frames each)
        #   stage 3: bbox + KL labels for all knees (1s)
        gt_frames: List[np.ndarray] = list(first_frames)
        cumulative = img.copy()
        for i, b in enumerate(boxes):
            color = KNEE_COLORS[i % len(KNEE_COLORS)]
            cumulative = draw_bbox(cumulative, b, color, label=sides[i])
            gt_frames.extend(loop_frames(cumulative, n=fps))
        final = overlay_all(img, boxes, grades, side_labels=sides)
        gt_frames.extend(loop_frames(final, n=fps))

        # Last segment: just the final annotated image (1s)
        last_frames = loop_frames(final, n=fps)

        return {
            "first_frame": first_frames[0],
            "final_frame": final,
            "first_frames": first_frames,
            "last_frames": last_frames,
            "gt_frames": gt_frames,
            "boxes": boxes,
            "grades": grades,
            "sides": sides,
            "image_id": raw["image_id"],
            "image_size": (h1, w1),
        }

    # ----------------------------------------------------- per-sample worker

    def process_sample(self, raw: dict, idx: int) -> Optional[TaskSample]:
        if not raw.get("bboxes_xyxy"):
            return None

        built = self._build_frames(raw)
        sid = f"{self.config.domain}_{idx:05d}"
        out = Path(self.config.output_dir) / f"{self.config.domain}_task" / sid
        out.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out / "first_frame.png"), built["first_frame"])
        cv2.imwrite(str(out / "final_frame.png"), built["final_frame"])

        fps = self.config.fps
        make_video(built["first_frames"], out / "first_video.mp4", fps)
        make_video(built["last_frames"], out / "last_video.mp4", fps)
        make_video(built["gt_frames"], out / "ground_truth.mp4", fps)

        # Prompt with GT
        gt_lines = []
        for side, grade, b in zip(built["sides"], built["grades"], built["boxes"]):
            kl_text = KL_GRADE_NAMES.get(grade, f"KL={grade}")
            gt_lines.append(
                f"  - {side} knee: bbox=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}) "
                f"KL grade={grade} ({kl_text})"
            )
        prompt = (
            f"{self.config.task_prompt}\n\n"
            f"Ground-truth findings:\n" + "\n".join(gt_lines) + "\n"
        )
        (out / "prompt.txt").write_text(prompt)

        meta = {
            "image_id": built["image_id"],
            "image_size_hw": list(built["image_size"]),
            "num_knees": len(built["boxes"]),
            "knees": [
                {"side": s, "bbox_xyxy": b, "kl_grade": int(g)}
                for s, b, g in zip(built["sides"], built["boxes"], built["grades"])
            ],
            "fps": fps,
            "frames_per_video": {
                "first": len(built["first_frames"]),
                "last": len(built["last_frames"]),
                "ground_truth": len(built["gt_frames"]),
            },
            "case_type": "D_localize_then_grade",
            "task_family": "knee_localization_kl_grading",
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

        return TaskSample(
            task_id=sid,
            domain=self.config.domain,
            prompt=prompt,
            first_image=built["first_frame"],
            final_image=built["final_frame"],
            first_video=str(out / "first_video.mp4"),
            last_video=str(out / "last_video.mp4"),
            ground_truth_video=str(out / "ground_truth.mp4"),
            metadata=meta,
        )

    def run(self) -> List[TaskSample]:
        out = []
        kept = 0
        for idx, raw in enumerate(self.download()):
            s = self.process_sample(raw, kept)
            if s is not None:
                out.append(s)
                kept += 1
        return out
