"""Pipeline configuration for M-121 (KneeXray KL grading + knee localization)."""
from pathlib import Path
from pydantic import Field
from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """Configuration for the KneeXray pipeline.

    Inherited from PipelineConfig:
        num_samples: Optional[int]
        domain: str
        output_dir: Path
        split: str
    """

    domain: str = Field(default="kneexray_kl_grading")

    s3_bucket: str = Field(
        default="med-vr-datasets",
        description="S3 bucket containing the raw KneeXray data",
    )
    s3_prefix: str = Field(
        default="M-121/kneexray/KneeXrayData/DetKneeData/H5/trainH5/",
        description=(
            "S3 prefix for the Detection-with-grading H5 files. Each .h5 holds "
            "one X-ray image, two knee bounding boxes, and two KL grades."
        ),
    )
    fps: int = Field(default=6)
    raw_dir: Path = Field(default=Path("raw"))

    task_prompt: str = Field(
        default=(
            "This is a frontal knee radiograph (X-ray). The video first shows "
            "the original image, then localises the LEFT and RIGHT knee joints "
            "with coloured bounding boxes, and finally annotates each joint "
            "with its Kellgren-Lawrence (KL) osteoarthritis grade (0=normal, "
            "1=doubtful, 2=mild, 3=moderate, 4=severe). Identify the bounding "
            "box coordinates and KL grade for each knee."
        ),
        description="Task instruction shown to the reasoning model.",
    )
