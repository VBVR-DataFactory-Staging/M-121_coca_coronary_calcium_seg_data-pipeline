"""Downloader for KneeXray DetKnee H5 dataset (M-121).

Each H5 file at ``raw/<id>.h5`` contains:
  - dataset ``images``  shape=(H, W, 3) dtype=float64 in [0, 255]
  - group   ``gt_boxes`` with sub-groups ``i0``, ``i1`` (one per knee)
            each sub-group's bbox is stored on its **HDF5 attrs** as
            ``i0, i1, i2, i3`` = (x1, y1, x2, y2) in pixel coords.
  - group   ``gt_classes`` whose attrs ``i0, i1`` are the KL grade
            integers (0..4) for the matching boxes.
  - root attrs ``origin_im`` is the OAI image stem.

The bootstrap v3 `aws s3 sync` populates ``raw/`` with these flat .h5
files before generate.py runs.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np


KL_GRADE_NAMES = {
    0: "Normal (KL 0)",
    1: "Doubtful (KL 1)",
    2: "Mild (KL 2)",
    3: "Moderate (KL 3)",
    4: "Severe (KL 4)",
}


def _read_h5(path: Path) -> Optional[dict]:
    """Parse one DetKnee .h5 file into a sample dict."""
    try:
        with h5py.File(path, "r") as f:
            if "images" not in f:
                return None
            img = f["images"][()]  # (H, W, 3) float64 in [0, 255]
            origin = f.attrs.get("origin_im", b"")
            if isinstance(origin, (bytes, bytearray)):
                origin = origin.decode("utf-8", errors="ignore")

            bboxes = []
            grades = []

            box_grp = f.get("gt_boxes")
            cls_grp = f.get("gt_classes")
            if box_grp is None or cls_grp is None:
                return None

            # gt_boxes is a "list:N" group with sub-groups i0, i1, ...
            box_keys = sorted(box_grp.keys(), key=lambda k: int(k[1:]))
            for k in box_keys:
                sub = box_grp[k]
                a = sub.attrs
                # deepdish stores list-of-floats as attrs i0..i3
                try:
                    coords = [float(a[f"i{j}"]) for j in range(4)]
                except KeyError:
                    continue
                bboxes.append(coords)  # (x1, y1, x2, y2)

            # gt_classes attrs i0, i1, ... are KL grade ints
            cls_attrs = cls_grp.attrs
            n = len(bboxes)
            for j in range(n):
                key = f"i{j}"
                if key in cls_attrs:
                    grades.append(int(cls_attrs[key]))
                else:
                    grades.append(-1)

        return {
            "image_id": origin or path.stem,
            "image": img,
            "bboxes_xyxy": bboxes,   # list of [x1, y1, x2, y2]
            "kl_grades": grades,     # list of ints 0..4
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[downloader] skip {path.name}: {exc}")
        return None


class TaskDownloader:
    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config.raw_dir)

    def iter_samples(self, limit: Optional[int] = None) -> Iterator[dict]:
        if not self.raw_dir.exists():
            print(f"[downloader] raw dir missing: {self.raw_dir}")
            return
        # raw/ may either be flat *.h5 files (after bootstrap sync of trainH5/)
        # or nested. We scan recursively for safety.
        h5_files = sorted(self.raw_dir.rglob("*.h5"))
        if not h5_files:
            print(f"[downloader] no .h5 files under {self.raw_dir}")
            return

        count = 0
        for p in h5_files:
            sample = _read_h5(p)
            if sample is None or not sample["bboxes_xyxy"]:
                continue
            yield sample
            count += 1
            if limit is not None and count >= limit:
                return

    def download(self, limit: Optional[int] = None) -> Iterator[dict]:
        yield from self.iter_samples(limit=limit)


def create_downloader(config) -> TaskDownloader:
    return TaskDownloader(config)
