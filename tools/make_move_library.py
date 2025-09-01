"""Create a move library from recorded moves cli utility."""

import argparse

from huggingface_hub import HfApi

parser = argparse.ArgumentParser(
    description="Create a move library from moves recorded with record_move.py, upload as a dataset to HuggingFace."
)
parser.add_argument(
    "-l", "--library", type=str, required=True, help="Path to the local moves library"
)
parser.add_argument(
    "--repo_id",
    type=str,
    required=True,
    help="HuggingFace repository ID (e.g., username/repo_name)",
)
parser.add_argument(
    "--public",
    action="store_true",
    default=False,
    help="Whether to make the dataset public",
)
args = parser.parse_args()


api = HfApi()

api.create_repo(
    exist_ok=True, repo_id=args.repo_id, repo_type="dataset", private=not args.public
)

api.upload_folder(
    folder_path=args.library,
    repo_id=args.repo_id,
    repo_type="dataset",
)
