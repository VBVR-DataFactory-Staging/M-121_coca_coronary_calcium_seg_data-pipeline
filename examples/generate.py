#!/usr/bin/env python3
"""Dataset generation entry point for M-121 KneeXray (KL grading + knee bbox).

Usage:
    python examples/generate.py
    python examples/generate.py --num-samples 10
    python examples/generate.py --output data/my_output
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TaskPipeline, TaskConfig


def main():
    parser = argparse.ArgumentParser(description="Generate M-121 KneeXray dataset")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="data/questions")
    args = parser.parse_args()

    print(f"Generating M-121 (kneexray_kl_grading) dataset — "
          f"num_samples={args.num_samples}, output={args.output}")
    config = TaskConfig(
        num_samples=args.num_samples,
        output_dir=Path(args.output),
    )
    pipeline = TaskPipeline(config)
    samples = pipeline.run()
    print(f"Done. Wrote {len(samples)} samples.")


if __name__ == "__main__":
    main()
