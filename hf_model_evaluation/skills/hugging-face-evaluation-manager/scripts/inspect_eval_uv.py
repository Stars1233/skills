# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "inspect-ai>=0.3.0",
#     "inspect-evals",
#     "openai",
# ]
# ///

"""
Entry point script for running inspect-ai evaluations via `hf jobs uv run`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect-ai job runner")
    parser.add_argument("--model", required=True, help="Model ID on Hugging Face Hub")
    parser.add_argument("--task", required=True, help="inspect-ai task to execute")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    args = parser.parse_args()

    # Ensure downstream libraries can read the token passed as a secret
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HF_HUB_TOKEN", hf_token)

    cmd = [
        "inspect",
        "eval",
        args.task,
        "--model",
        f"hf-inference-providers/{args.model}",
        "--log-level",
        "info",
        # Reduce batch size to avoid OOM errors (default is 32)
        "--max-connections",
        "1",
        # Set a small positive temperature (HF doesn't allow temperature=0)
        "--temperature",
        "0.001",
    ]

    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    try:
        subprocess.run(cmd, check=True)
        print("Evaluation complete.")
    except subprocess.CalledProcessError as exc:
        print(f"Evaluation failed with exit code {exc.returncode}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
