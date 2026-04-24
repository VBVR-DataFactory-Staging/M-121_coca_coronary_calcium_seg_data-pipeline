"""M-121: FLARE22 progressive organ-by-organ segmentation reveal.

Distinct from M-111 (simultaneous 13-organ co-segmentation): here the
reasoning task is *temporal organ ordering*. The ground_truth video shows
organs being added to the overlay one at a time, in a fixed anatomical
sequence (large → small, central → peripheral). The model must infer
which organ is next.

Dataset: FLARE 2022 challenge training set (50 abdominal CT volumes).
Raw layout (after extract to _extracted/M-111_FLARE22/):
    FLARE22Train/
        images/FLARE22_Tr_XXXX_0000.nii.gz
        labels/FLARE22_Tr_XXXX.nii.gz

FLARE22 label index → organ:
    1=liver, 2=right kidney, 3=spleen, 4=pancreas, 5=aorta, 6=IVC,
    7=right adrenal, 8=left adrenal, 9=gallbladder, 10=esophagus,
    11=stomach, 12=duodenum, 13=left kidney.

Video construction (Case B variant — slice sweep PLUS organ-reveal sweep):
- first_video.mp4: raw axial sweep (no overlay)
- last_video.mp4:  axial sweep with all 13 organs overlaid simultaneously
- ground_truth.mp4: STATIC representative slice, organs revealed
  one-by-one in the canonical anatomical sequence (1 organ added per
  ~3 frames at FPS=3, total ~13*3=39 frames per case + axial recap).
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import nibabel as nib
from common import (
    DATA_ROOT, window_ct, to_rgb, overlay_multi, write_task,
    COLORS, fit_square, pick_annotated_idx,
)

PID = "M-121"
TASK_NAME = "flare22_progressive_organ_reveal"
FPS = 3

# Canonical anatomical sequence: large central organs first, then
# peripheral / small structures. Each entry: (label_value, name, color).
# This ordering is what the model needs to learn.
ORGAN_SEQUENCE = [
    (1,  "liver",               "green"),
    (3,  "spleen",              "purple"),
    (11, "stomach",             "yellow"),
    (2,  "right_kidney",        "orange"),
    (13, "left_kidney",         "orange"),
    (4,  "pancreas",            "pink"),
    (5,  "aorta",               "red"),
    (6,  "IVC",                 "blue"),
    (12, "duodenum",            "brown"),
    (10, "esophagus",           "cyan"),
    (9,  "gallbladder",         "lime"),
    (7,  "right_adrenal_gland", "teal"),
    (8,  "left_adrenal_gland",  "teal"),
]

# Build a remapped label list for overlay_multi (uses sequential 1..N indexing).
# We create a per-step label_map to feed overlay_multi cumulatively.

PROMPT = (
    "This is an abdominal CT scan from the FLARE 2022 dataset. The video "
    "reveals abdominal organs ONE AT A TIME, in a fixed anatomical sequence "
    "from largest/central to smallest/peripheral. Predict the order in which "
    "the 13 organs are added to the overlay and name each newly revealed "
    "organ at every reveal step. Organs in the dataset: liver, spleen, "
    "stomach, right kidney, left kidney, pancreas, aorta, inferior vena "
    "cava (IVC), duodenum, esophagus, gallbladder, right adrenal gland, "
    "left adrenal gland."
)


def best_slice_idx(lbl_vol: np.ndarray) -> int:
    """Pick the axial slice that contains the most distinct organ labels —
    used as the still-frame backdrop for the progressive reveal video."""
    best_z, best_count = 0, -1
    for z in range(lbl_vol.shape[0]):
        present = np.unique(lbl_vol[z])
        present = present[present > 0]
        if len(present) > best_count:
            best_count = len(present)
            best_z = z
    return best_z


def build_progressive_frames(rgb_slice: np.ndarray, lbl_slice: np.ndarray,
                             frames_per_step: int = 3):
    """Reveal organs one-by-one on a single slice. Returns list of BGR frames."""
    frames = []
    cumulative = np.zeros_like(lbl_slice, dtype=np.int32)
    seq_idx = 0
    for orig_label, name, color in ORGAN_SEQUENCE:
        present = (lbl_slice == orig_label)
        if present.any():
            seq_idx += 1
            cumulative[present] = seq_idx
        # build color list for everything revealed so far (sequential 1..seq_idx)
        color_list = []
        i = 0
        for ol, nm, c in ORGAN_SEQUENCE:
            if (lbl_slice == ol).any():
                i += 1
                if i <= seq_idx:
                    color_list.append((nm, COLORS[c]))
        ann = overlay_multi(rgb_slice, cumulative, color_list)
        for _ in range(frames_per_step):
            frames.append(ann)
    return frames


def process_case(img_path: Path, lbl_path: Path, task_idx: int):
    img_vol = np.transpose(nib.load(str(img_path)).get_fdata(), (2, 1, 0))
    lbl_vol = np.transpose(nib.load(str(lbl_path)).get_fdata(), (2, 1, 0)).astype(np.int32)

    n = img_vol.shape[0]
    step = max(1, n // 60)
    indices = list(range(0, n, step))[:60]

    # Build all-organs color list for last_video (uses re-mapped 1..K labels)
    # We remap label_vol so each present organ gets a sequential id matching
    # ORGAN_SEQUENCE order — so overlay_multi colors match.
    remap = np.zeros_like(lbl_vol, dtype=np.int32)
    full_color_list = []
    seq = 0
    for orig_label, name, color in ORGAN_SEQUENCE:
        if (lbl_vol == orig_label).any():
            seq += 1
            remap[lbl_vol == orig_label] = seq
            full_color_list.append((name, COLORS[color]))

    first_frames, last_frames, flags = [], [], []
    for z in indices:
        ct = window_ct(img_vol[z])
        rgb = to_rgb(ct)
        rgb = fit_square(rgb, 512)
        lab_square = fit_square(remap[z].astype(np.int16), 512).astype(np.int32)
        ann = overlay_multi(rgb, lab_square, full_color_list)
        first_frames.append(rgb)
        last_frames.append(ann)
        flags.append(bool((lab_square > 0).any()))

    pick = pick_annotated_idx(flags)
    first_frame = first_frames[pick]
    final_frame = last_frames[pick]

    # Progressive reveal — uses the slice with most organs
    bz = best_slice_idx(lbl_vol)
    bz_rgb = fit_square(to_rgb(window_ct(img_vol[bz])), 512)
    bz_lbl = fit_square(lbl_vol[bz].astype(np.int16), 512).astype(np.int32)
    gt_frames = build_progressive_frames(bz_rgb, bz_lbl, frames_per_step=3)
    if not gt_frames:
        gt_frames = last_frames[:5]

    organs_present = sorted({int(v) for v in np.unique(lbl_vol) if v > 0})
    organ_names_present = [n for ol, n, _ in ORGAN_SEQUENCE if ol in organs_present]

    meta = {
        "task": "FLARE22 progressive organ-by-organ reveal",
        "dataset": "FLARE 2022",
        "case_id": img_path.stem.replace("_0000", "").replace(".nii", ""),
        "modality": "CT",
        "task_type": "temporal organ ordering / progressive reveal",
        "reveal_sequence": [n for _, n, _ in ORGAN_SEQUENCE],
        "organs_present": organ_names_present,
        "colors": {n: c for _, n, c in ORGAN_SEQUENCE},
        "fps_source": "case B progressive reveal, 3 frames per organ-add",
        "num_slices_total": int(n),
        "num_slices_axial_sweep": len(indices),
        "num_organs_revealed": len(organ_names_present),
        "key_slice_z": int(bz),
        "source_split": "train",
    }
    return write_task(PID, TASK_NAME, task_idx,
                      first_frame, final_frame,
                      first_frames, last_frames, gt_frames,
                      PROMPT, meta, FPS)


def main():
    root = DATA_ROOT / "_extracted" / "M-111_FLARE22" / "FLARE22Train"
    cases = sorted(root.glob("images/FLARE22_Tr_*_0000.nii.gz"))
    print(f"  {len(cases)} FLARE22 cases (M-121 progressive variant)")
    for i, img in enumerate(cases):
        label_name = img.name.replace("_0000.nii.gz", ".nii.gz")
        lbl = root / "labels" / label_name
        if not lbl.exists():
            print(f"  skip {img.name}: no label")
            continue
        try:
            d = process_case(img, lbl, i)
            print(f"  wrote {d}")
        except Exception as e:
            print(f"  ERROR on {img.name}: {e}")


if __name__ == "__main__":
    main()
