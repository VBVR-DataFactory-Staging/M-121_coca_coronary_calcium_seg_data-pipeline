# M-121 — FLARE22 Progressive Organ-by-Organ Reveal

FLARE 2022 abdominal CT 13-organ progressive segmentation reveal task.

This repository is part of the Med-VR data-pipeline suite for the VBVR
(Very Big Video Reasoning) benchmark. It produces standardized video-
reasoning task samples from the underlying raw medical dataset.

## Task

**Distinct from M-111 (simultaneous 13-organ co-segmentation):** here the
reasoning task is *temporal organ ordering*. The ground-truth video shows
abdominal organs being added to the segmentation overlay one at a time, in
a fixed anatomical sequence (large/central → small/peripheral). The model
must predict the order and name each newly revealed organ.

**Prompt shown to the model**:

> This is an abdominal CT scan from the FLARE 2022 dataset. The video
> reveals abdominal organs ONE AT A TIME, in a fixed anatomical sequence
> from largest/central to smallest/peripheral. Predict the order in which
> the 13 organs are added to the overlay and name each newly revealed
> organ at every reveal step. Organs in the dataset: liver, spleen,
> stomach, right kidney, left kidney, pancreas, aorta, inferior vena
> cava (IVC), duodenum, esophagus, gallbladder, right adrenal gland,
> left adrenal gland.

## Reveal Sequence (canonical anatomical order)

1. liver, 2. spleen, 3. stomach, 4. right kidney, 5. left kidney,
6. pancreas, 7. aorta, 8. IVC, 9. duodenum, 10. esophagus,
11. gallbladder, 12. right adrenal gland, 13. left adrenal gland.

## S3 Raw Data

```
s3://med-vr-datasets/_extracted/M-111_FLARE22/FLARE22Train/
    images/FLARE22_Tr_XXXX_0000.nii.gz   (50 CT volumes)
    labels/FLARE22_Tr_XXXX.nii.gz        (13-organ masks)
```

(Shared raw with M-111; the two pipelines differ only in prompt + video
construction.)

## Quick Start

```bash
pip install -r requirements.txt

# Generate samples (downloads raw from S3 on first run)
python examples/generate.py

# Generate only N samples
python examples/generate.py --num-samples 10

# Custom output directory
python examples/generate.py --output data/my_output
```

## Output Layout

```
data/questions/flare22_progressive_organ_reveal_task/
├── task_0000/
│   ├── first_frame.png        ← representative axial slice (raw)
│   ├── final_frame.png        ← same slice, all organs overlaid
│   ├── first_video.mp4        ← raw axial sweep
│   ├── last_video.mp4         ← axial sweep with all 13 organs
│   ├── ground_truth.mp4       ← progressive organ-by-organ reveal
│   ├── prompt.txt
│   └── metadata.json
├── task_0001/
└── ...
```

## Configuration

`src/pipeline/config.py` (`TaskConfig`):

| Field | Default | Description |
|---|---|---|
| `domain` | `"flare22_progressive_organ_reveal"` | Task domain string used in output paths. |
| `s3_bucket` | `"med-vr-datasets"` | S3 bucket containing raw data. |
| `s3_prefix` | `"_extracted/M-111_FLARE22/"` | S3 key prefix for raw data (shared with M-111). |
| `fps` | `3` | Output video FPS. |
| `raw_dir` | `Path("raw")` | Local raw cache directory. |
| `num_samples` | `None` | Max samples (None = all). |

## Repository Structure

```
M-121_coca_coronary_calcium_seg_data-pipeline/   ← repo name kept for URL stability
├── core/                ← shared pipeline framework (verbatim)
├── eval/                ← shared evaluation utilities
├── src/
│   ├── download/
│   │   └── downloader.py   ← S3 raw-data downloader
│   └── pipeline/
│       ├── config.py        ← task config
│       ├── pipeline.py      ← TaskPipeline
│       ├── transforms.py    ← visualization helpers (shim)
│       └── _phase2/
│           ├── common.py
│           └── m121_flare22_progressive.py   ← core logic
├── examples/
│   └── generate.py
├── requirements.txt
├── README.md
└── LICENSE
```

> **Note on repo name**: the GitHub repository is still called
> `M-121_coca_coronary_calcium_seg_data-pipeline` because the website
> URL is keyed on this slug. The actual task is FLARE22 progressive
> organ reveal, as documented above.
