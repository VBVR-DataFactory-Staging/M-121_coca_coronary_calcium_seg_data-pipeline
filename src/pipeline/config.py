"""Pipeline configuration for M-121 (flare22_progressive_organ_reveal)."""
from pathlib import Path
from pydantic import Field
from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """Configuration for M-121 pipeline.

    Inherited from PipelineConfig:
        num_samples: Optional[int]  # Max samples (None = all)
        domain: str
        output_dir: Path
        split: str
    """

    domain: str = Field(default="flare22_progressive_organ_reveal")

    s3_bucket: str = Field(
        default="med-vr-datasets",
        description="S3 bucket containing the raw FLARE22 data",
    )
    s3_prefix: str = Field(
        default="_extracted/M-111_FLARE22/",
        description="S3 key prefix for the dataset raw data (shared with M-111)",
    )
    fps: int = Field(
        default=3,
        description="Frames per second for the generated videos",
    )
    raw_dir: Path = Field(
        default=Path("raw"),
        description="Local directory for downloaded raw data",
    )
    task_prompt: str = Field(
        default=(
            "Abdominal CT (FLARE22). Organs are revealed one-by-one in a "
            "fixed anatomical sequence (large/central → small/peripheral). "
            "Predict the next organ in the reveal sequence."
        ),
        description="The task instruction shown to the reasoning model.",
    )
