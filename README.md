# M-121 — KneeXray Knee Localization + KL Grading

Knee X-ray task: detect and localize the LEFT and RIGHT knee joint, then
assign each joint a Kellgren-Lawrence (KL) osteoarthritis grade (0–4).

This repository is part of the Med-VR data-pipeline suite for the VBVR
(Very Big Video Reasoning) benchmark. It produces standardized 7-file
video reasoning samples from the OAI-derived **KneeXrayData** dataset
(Mendeley `56rmx5bjcr`).

> Repository name kept as the original `M-121_coca_coronary_calcium_seg_data-pipeline`
> slot — the website maps task pages by repo prefix and we are filling
> this previously-empty slot with the new KneeXray task.

## Task

**Prompt shown to the model**:

> This is a frontal knee radiograph (X-ray). The video first shows the
> original image, then localises the LEFT and RIGHT knee joints with
> coloured bounding boxes, and finally annotates each joint with its
> Kellgren-Lawrence (KL) osteoarthritis grade (0=normal, 1=doubtful,
> 2=mild, 3=moderate, 4=severe). Identify the bounding box coordinates
> and KL grade for each knee.

## Sample layout

Each sample writes the standard 7-file VBVR layout under
`data/questions/kneexray_kl_grading_task/kneexray_kl_grading_NNNNN/`:

| file              | contents                                            |
|-------------------|-----------------------------------------------------|
| `first_frame.png` | raw knee X-ray (upscaled to 640px short side)       |
| `final_frame.png` | image with both knee bboxes + KL grade text overlay |
| `first_video.mp4` | 1 s loop of the raw image (fps=6)                   |
| `last_video.mp4`  | 1 s loop of the fully annotated image               |
| `ground_truth.mp4`| progressive reveal: plain → bbox-only → bbox+grade  |
| `prompt.txt`      | task instruction + ground-truth bbox & KL listing   |
| `metadata.json`   | image_id, image_size, knees[{side,bbox_xyxy,grade}] |

## Raw data

S3 location (already populated): `s3://med-vr-datasets/M-121/kneexray/`

```
KneeXrayData/
  ClsKLData/kneeKL224/{train,test,val,auto_test}/{0..4}/*.png
  DetKneeData/H5/{train,test,val}H5/*.h5
  DetKneeData/best_models/
```

This pipeline consumes **DetKneeData/H5/trainH5/** (2,889 H5 files) — each
file packs one full A/P knee radiograph with two bounding boxes (one per
knee) and the per-knee KL grade as deepdish-style HDF5 attrs.

## Local generation

```bash
pip install -r requirements.txt
# Sync raw/ (or rely on the EC2 bootstrap v3 auto-sync from s3_prefix):
aws s3 sync s3://med-vr-datasets/M-121/kneexray/KneeXrayData/DetKneeData/H5/trainH5/ raw/
python examples/generate.py --num-samples 5 --output /tmp/m121-test
```

## Cluster generation

This repo is launched by the standard Med-VR EC2 bootstrap (c5.4xlarge,
self-terminating). The v3 user-data block syncs `raw/` from the S3 prefix
declared in `src/pipeline/config.py` before invoking
`examples/generate.py`. Output is then pushed to
`s3://vbvr-final-data/questions/M-121_coca_coronary_calcium_seg_data-pipeline/`
and surfaces on https://vm-dataset.com.
