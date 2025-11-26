#!/usr/bin/env python3
"""
Create MP4 animation from 3D validation plots.
Modify the configuration variables below as needed.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# Configuration - modify these values as needed
ROBOT_TYPE = "Isaac_Jetbot"
CONFIDENCE_LEVEL = "over_confident"
VALIDATION_ID = "0336"
BOUNDARY_TYPE = "boundary_5000"
OUTPUT_VIDEO = "3d_rotation.mp4"
FPS = 16  # frames per second
QUALITY = "high"  # high, medium, low


def main():
    # Build paths
    base_path = (
        Path("data")
        / ROBOT_TYPE
        / "reports"
        / CONFIDENCE_LEVEL
        / BOUNDARY_TYPE
        / "plots_3d"
    )

    print(f"Looking for 3D plots in: {base_path}")

    if not base_path.exists():
        print(f"Error: Directory {base_path} does not exist")
        return

    # Find all PNG files for this validation ID
    pattern = f"fig_3D_*{VALIDATION_ID}_az*.png"
    png_files = list(base_path.glob(pattern))

    if not png_files:
        print(f"Error: No PNG files found matching pattern {pattern}")
        return

    print(f"Found {len(png_files)} PNG files")

    # Extract azimuth angles and sort
    azimuth_files = []
    for file in png_files:
        match = re.search(r"az(\d{3})\.png$", file.name)
        if match:
            azimuth = int(match.group(1))
            azimuth_files.append((azimuth, file))

    # Sort by azimuth for smooth rotation (0-360 degrees)
    azimuth_files.sort(key=lambda x: x[0])

    print(f"Found {len(azimuth_files)} azimuth angles")
    print("Files in azimuth order:")
    for azimuth, file in azimuth_files:
        print(f"  {azimuth:03d}°: {file.name}")

    # Create temporary directory for numbered files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy files to temp directory with sequential numbering
        for i, (azimuth, file) in enumerate(azimuth_files):
            temp_file = temp_path / f"frame_{i:04d}.png"
            shutil.copy2(file, temp_file)
            print(f"Copied {file.name} -> {temp_file.name}")

        # Set video quality parameters
        if QUALITY == "high":
            bitrate = "5000k"
            preset = "slow"
        elif QUALITY == "medium":
            bitrate = "2000k"
            preset = "medium"
        else:  # low
            bitrate = "1000k"
            preset = "fast"

        # Create output filename with descriptive info
        output_path = f"{ROBOT_TYPE}_{CONFIDENCE_LEVEL}_{VALIDATION_ID}_3D_rotation.mp4"

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # overwrite output file
            "-framerate",
            str(FPS),
            "-i",
            str(temp_path / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-b:v",
            bitrate,
            "-pix_fmt",
            "yuv420p",  # for compatibility
            output_path,
        ]

        print(f"\nCreating video: {output_path}")
        print(f"Quality: {QUALITY}, FPS: {FPS}, Bitrate: {bitrate}")

        # Run ffmpeg
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Successfully created: {output_path}")

                # Show file info
                if os.path.exists(output_path):
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"  File size: {size_mb:.1f} MB")

            else:
                print("✗ Error creating video:")
                print(result.stderr)

        except FileNotFoundError:
            print("Error: ffmpeg not found. Please install ffmpeg.")
        except Exception as e:
            print(f"Error running ffmpeg: {e}")


if __name__ == "__main__":
    main()
