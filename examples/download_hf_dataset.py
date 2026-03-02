#!/usr/bin/env python3
"""Download dataset from Hugging Face Hub."""

from huggingface_hub import snapshot_download
import pathlib

# Download the dataset
repo_id = "Facebear/XVLA-Soft-Fold"
local_dir = pathlib.Path("/data0/ygx_data/X-VLA")
allow_patterns = [
    "0930_10am_new/*",
    "README.md",
    "camera_angle.png",
    "camera_mount.SLDPRT",
    "camera_mount.STEP",
    "camera_mount.STL",
    "camera_mount_install.md",
    "overview_camera_mount.png",
    "the_main_view.png"
]

print(f"Downloading {repo_id} to {local_dir}...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=allow_patterns,
    local_dir_use_symlinks=False
)

print("Download complete!")
