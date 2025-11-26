#!/usr/bin/env python3
"""
Process Real MBot Data for CLAPS Pipeline

Converts LCM log files from real MBot experiments into CLAPS-compatible .pt files
for calibration and validation. 

Usage:
    python scripts/process_real_mbot_data.py --log_files path/to/log1.lcm [path/to/log2.lcm ...]
    python scripts/process_real_mbot_data.py --log_dir path/to/directory/with/logs/
"""

import argparse
import glob
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add paths for imports from external/mbot_experiments
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)  # claps_code root
mbot_scripts_dir = os.path.join(
    workspace_root, "external", "mbot_experiments", "scripts"
)

if mbot_scripts_dir not in sys.path:
    sys.path.insert(0, mbot_scripts_dir)

mbot_root = os.path.join(workspace_root, "external", "mbot_experiments")
if mbot_root not in sys.path:
    sys.path.insert(0, mbot_root)

from scripts.config import DEFAULT_POSE_CHANNEL, VICON_BODY_NAME

from scripts.exp_utils import (
    extract_motion_segments,
    import_lcm_messages,
    read_lcm_experiment_log,
    transform_vicon_to_centroid,
)

try:
    import_lcm_messages()
except ImportError as e:
    print(
        f"⚠️ Warning: Could not import LCM messages. Some functionality may be limited. Error: {e}"
    )
    import traceback
    traceback.print_exc()

# Safe workspace bounds based on observed Real MBot data (relaxed to avoid cutting off valid data)
SAFE_WORKSPACE_BOUNDS = {
    "x_min": 0.18,
    "x_max": 1.42,
    "y_min": 0.0,
    "y_max": 2.1,
}

def setup_logging(output_dir: str, verbose: bool = True) -> logging.Logger:
    """Set up comprehensive logging to file and console."""
    # Create logs directory in project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)
    logs_dir = os.path.join(workspace_root, "logs", "real_mbot_processing")
    os.makedirs(logs_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"process_real_mbot_data_{timestamp}.log")

    # Configure logging
    logger = logging.getLogger("process_real_mbot_data")
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def analyze_velocity_distribution(
    raw_segments: List[Dict],
    accepted_claps_data: List[Dict],
    velocity_bounds: Dict,
    output_dir: str,
    logger: logging.Logger,
):
    """Analyze and visualize velocity distributions before and after filtering."""
    logger.info("Analyzing velocity distributions before and after filtering...")

    # Extract velocities from RAW segments (before filtering)
    raw_vx_start, raw_vx_end = [], []
    raw_wz_start, raw_wz_end = [], []

    for segment in raw_segments:
        measured_twists = segment.get("measured_twists", [])
        if len(measured_twists) >= 2:
            first_twist = measured_twists[0]
            last_twist = measured_twists[-1]

            raw_vx_start.append(first_twist[1])  # vx at start
            raw_vx_end.append(last_twist[1])  # vx at end
            raw_wz_start.append(abs(first_twist[3]))  # |wz| at start
            raw_wz_end.append(abs(last_twist[3]))  # |wz| at end

    # Extract velocities from ACCEPTED CLAPS data (after filtering)
    # Note: CLAPS data contains world-frame velocities, we need to convert back for comparison
    accepted_vx_start, accepted_vx_end = [], []
    accepted_wz_start, accepted_wz_end = [], []

    for claps_segment in accepted_claps_data:
        # Extract angular velocities (same in both frames)
        wz0 = claps_segment["dq0"][2].item()
        wz1 = claps_segment["dq1"][2].item()
        accepted_wz_start.append(abs(wz0))
        accepted_wz_end.append(abs(wz1))

        # For linear velocities, we use magnitude of velocity vector
        vx0 = claps_segment["dq0"][0].item()
        vy0 = claps_segment["dq0"][1].item()
        vx1 = claps_segment["dq1"][0].item()
        vy1 = claps_segment["dq1"][1].item()

        v_mag_0 = np.sqrt(vx0**2 + vy0**2)
        v_mag_1 = np.sqrt(vx1**2 + vy1**2)
        accepted_vx_start.append(v_mag_0)
        accepted_vx_end.append(v_mag_1)

    # Combine all velocities for analysis
    raw_vx = np.array(raw_vx_start + raw_vx_end)
    raw_wz = np.array(raw_wz_start + raw_wz_end)
    accepted_vx = np.array(accepted_vx_start + accepted_vx_end)
    accepted_wz = np.array(accepted_wz_start + accepted_wz_end)

    logger.info(f"Raw (before filtering): {len(raw_vx)} velocity samples")
    logger.info(f"Accepted (after filtering): {len(accepted_vx)} velocity samples")
    logger.info(
        f"Filtered out: {len(raw_vx) - len(accepted_vx)} samples ({100*(len(raw_vx) - len(accepted_vx))/len(raw_vx):.1f}%)"
    )

    # Get velocity bounds
    linear_bounds = velocity_bounds["linear_bounds"]  # [min_vx, max_vx]
    angular_bounds = velocity_bounds["angular_bounds"]  # [min_wz, max_wz]

    # Count violations in raw data
    raw_vx_violations = np.sum(
        (raw_vx < linear_bounds[0]) | (raw_vx > linear_bounds[1])
    )
    raw_wz_violations = np.sum(raw_wz > angular_bounds[1])

    # Count violations in accepted data (should be 0 or very few)
    accepted_vx_violations = np.sum(
        (accepted_vx < linear_bounds[0]) | (accepted_vx > linear_bounds[1])
    )
    accepted_wz_violations = np.sum(accepted_wz > angular_bounds[1])

    logger.info(f"Velocity bound violations:")
    logger.info(f"  Linear bounds: [{linear_bounds[0]}, {linear_bounds[1]}] m/s")
    logger.info(f"  Angular bound: {angular_bounds[1]} rad/s")
    logger.info(
        f"  Raw data violations: {raw_vx_violations + raw_wz_violations}/{len(raw_vx)} ({100*(raw_vx_violations + raw_wz_violations)/len(raw_vx):.1f}%)"
    )
    logger.info(
        f"  Accepted data violations: {accepted_vx_violations + accepted_wz_violations}/{len(accepted_vx)} ({100*(accepted_vx_violations + accepted_wz_violations)/max(1,len(accepted_vx)):.1f}%)"
    )

    # Create 2x2 subplot: Before/After filtering for Linear/Angular
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Top row: BEFORE filtering
    # Linear velocity histogram (before)
    ax1.hist(
        raw_vx,
        bins=50,
        alpha=0.7,
        color="red",
        edgecolor="black",
        label="Before filtering",
    )
    ax1.axvline(
        linear_bounds[0],
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"Min bound: {linear_bounds[0]:.2f}",
    )
    ax1.axvline(
        linear_bounds[1],
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"Max bound: {linear_bounds[1]:.2f}",
    )
    ax1.set_xlabel("Linear Velocity vx (m/s)")
    ax1.set_ylabel("Count")
    ax1.set_title(
        f"Linear Velocity - BEFORE Filtering\n{raw_vx_violations} violations out of {len(raw_vx)} samples ({100*raw_vx_violations/len(raw_vx):.1f}%)"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Angular velocity histogram (before)
    ax2.hist(
        raw_wz,
        bins=50,
        alpha=0.7,
        color="orange",
        edgecolor="black",
        label="Before filtering",
    )
    ax2.axvline(
        angular_bounds[1],
        color="darkorange",
        linestyle="--",
        linewidth=2,
        label=f"Max bound: {angular_bounds[1]:.2f}",
    )
    ax2.set_xlabel("Angular Velocity |wz| (rad/s)")
    ax2.set_ylabel("Count")
    ax2.set_title(
        f"Angular Velocity - BEFORE Filtering\n{raw_wz_violations} violations out of {len(raw_wz)} samples ({100*raw_wz_violations/len(raw_wz):.1f}%)"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom row: AFTER filtering
    # Linear velocity histogram (after)
    if len(accepted_vx) > 0:
        ax3.hist(
            accepted_vx,
            bins=50,
            alpha=0.7,
            color="green",
            edgecolor="black",
            label="After filtering",
        )
        ax3.axvline(
            linear_bounds[0],
            color="darkgreen",
            linestyle="--",
            linewidth=2,
            label=f"Min bound: {linear_bounds[0]:.2f}",
        )
        ax3.axvline(
            linear_bounds[1],
            color="darkgreen",
            linestyle="--",
            linewidth=2,
            label=f"Max bound: {linear_bounds[1]:.2f}",
        )
        ax3.set_title(
            f"Linear Velocity - AFTER Filtering\n{accepted_vx_violations} violations out of {len(accepted_vx)} samples ({100*accepted_vx_violations/len(accepted_vx):.1f}%)"
        )
    else:
        ax3.set_title("Linear Velocity - AFTER Filtering\nNo data remaining")
    ax3.set_xlabel("Linear Velocity |v| (m/s)")
    ax3.set_ylabel("Count")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Angular velocity histogram (after)
    if len(accepted_wz) > 0:
        ax4.hist(
            accepted_wz,
            bins=50,
            alpha=0.7,
            color="blue",
            edgecolor="black",
            label="After filtering",
        )
        ax4.axvline(
            angular_bounds[1],
            color="darkblue",
            linestyle="--",
            linewidth=2,
            label=f"Max bound: {angular_bounds[1]:.2f}",
        )
        ax4.set_title(
            f"Angular Velocity - AFTER Filtering\n{accepted_wz_violations} violations out of {len(accepted_wz)} samples ({100*accepted_wz_violations/len(accepted_wz):.1f}%)"
        )
    else:
        ax4.set_title("Angular Velocity - AFTER Filtering\nNo data remaining")
    ax4.set_xlabel("Angular Velocity |wz| (rad/s)")
    ax4.set_ylabel("Count")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "reports", "velocity_filtering_analysis.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # Explicitly close the figure to free memory
    logger.info(f"Velocity filtering analysis saved to: {plot_path}")

    return


def find_vicon_state_data(log_file: str) -> List[Tuple]:
    """
    Extract Vicon state data from LCM log for pose extraction.

    Args:
        log_file: Path to LCM log file

    Returns:
        List of (timestamp, vicon_state_msg) tuples
    """
    # Import required message types
    try:
        from vicon_msgs.vicon_state_t import vicon_state_t
    except ImportError:
        # Try alternative import path
        try:
            from external.vicon2lcm.vicon_msgs.vicon_state_t import vicon_state_t
        except ImportError:
            print(
                "❌ Cannot import vicon_state_t. Make sure you're running on desktop with proper paths."
            )
            return []

    vicon_data = []

    try:
        import lcm

        log = None
        log = lcm.EventLog(log_file, "r")

        for event in log:
            if event.channel == DEFAULT_POSE_CHANNEL:  # Use constant from config
                msg = vicon_state_t.decode(event.data)
                vicon_data.append((event.timestamp, msg))

        print(f"📍 Extracted {len(vicon_data)} Vicon state messages from {log_file}")
        return vicon_data

    except Exception as e:
        print(f"❌ Error reading Vicon data from {log_file}: {e}")
        return []
    finally:
        if log is not None:
            log.close()

def is_pose_in_safe_workspace(x: float, y: float) -> bool:
    """Check if pose is within safe workspace bounds."""
    return (
        SAFE_WORKSPACE_BOUNDS["x_min"] <= x <= SAFE_WORKSPACE_BOUNDS["x_max"]
        and SAFE_WORKSPACE_BOUNDS["y_min"] <= y <= SAFE_WORKSPACE_BOUNDS["y_max"]
    )

def extract_segment_poses(
    vicon_data: List[Tuple],
    segment: Dict,
    tolerance_ms: float = 500.0,
    vicon_body_name: str = VICON_BODY_NAME,
) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """
    Extract start and end poses for a segment.

    Returns:
        (start_pose, end_pose): Tuples of (x, y, theta) or None if not found
    """
    if not vicon_data:
        return None, None

    start_time = segment["start_time"]
    end_time = segment["end_time"]
    tolerance_us = tolerance_ms * 1000  # Convert to microseconds

    def find_closest_pose(target_time):
        best_match = None
        min_time_diff = float("inf")

        for vicon_time, vicon_msg in vicon_data:
            time_diff = abs(vicon_time - target_time)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                best_match = (vicon_time, vicon_msg)

        if best_match is None or min_time_diff > tolerance_us:
            return None

        _, vicon_msg = best_match

        # Find robot body
        robot_idx = -1
        for i in range(int(vicon_msg.num_bodies)):
            if vicon_body_name.lower() in vicon_msg.body_names[i].lower():
                robot_idx = i
                break

        if robot_idx == -1:
            return None

        # Extract pose and transform to centroid
        vicon_x = float(vicon_msg.positions[robot_idx][0])
        vicon_y = float(vicon_msg.positions[robot_idx][1])
        yaw = float(vicon_msg.rpy[robot_idx][2])

        centroid_x, centroid_y = transform_vicon_to_centroid(vicon_x, vicon_y, yaw)
        return centroid_x, centroid_y, yaw

    start_pose = find_closest_pose(start_time)
    end_pose = find_closest_pose(end_time)

    return start_pose, end_pose


def validate_segment_quality(segment: Dict) -> Tuple[bool, List[str]]:
    """
    Returns:
        (is_valid, issues): Boolean validity and list of issue descriptions
    """
    issues = []

    # Check basic properties
    if segment["status"] != "COMPLETED":
        issues.append(f"Non-completed status: {segment['status']}")

    if segment["actual_duration"] < 0.05:
        issues.append(f"Too short: {segment['actual_duration']:.3f}s")

    # Additional check for segments that are too short compared to expected duration
    expected_duration = segment.get("command", {}).get("duration", 0.5)
    if (
        segment["actual_duration"] < expected_duration * 0.3
    ):  # Less than 30% of expected
        issues.append(
            f"Severely truncated: {segment['actual_duration']:.3f}s vs expected {expected_duration:.3f}s"
        )

    if len(segment.get("measured_twists", [])) < 3:
        issues.append(
            f"Insufficient twist data: {len(segment.get('measured_twists', []))} points"
        )

    # Check command validity
    command = segment.get("command", {})
    ax = command.get("ax", 0)
    az = command.get("az", 0)

    # Constant velocity segments (ax=0, az=0) are VALID
    # Only filter out segments with truly invalid commands
    if not (np.isfinite(ax) and np.isfinite(az)):
        issues.append("Invalid commanded acceleration (NaN or inf)")

    # Check Vicon data quality
    vicon_count = segment.get("vicon_points", len(segment.get("measured_twists", [])))
    expected_points = segment["actual_duration"] * 100  # 100 Hz expected

    if vicon_count < expected_points * 0.6:  # Less than 60% expected data
        issues.append(
            f"Low Vicon data rate: {vicon_count} vs {expected_points:.0f} expected"
        )

    # Check duration consistency - compare actual vs commanded duration
    expected_duration = command.get("duration", 0.5)  # Default 0.5s if not specified
    duration_error = abs(segment["actual_duration"] - expected_duration)
    duration_error_percent = duration_error / expected_duration * 100

    # Flag segments with >20% duration error or >80ms absolute error (realistic for 25Hz control: 2 periods)
    if duration_error > max(0.08, expected_duration * 0.2):
        issues.append(
            f"Duration error: actual={segment['actual_duration']:.4f}s vs expected={expected_duration:.4f}s ({duration_error_percent:.1f}% error)"
        )

    is_valid = len(issues) == 0
    return is_valid, issues


def process_log_file(
    args: Tuple[str, List[float], List[float]],
) -> Tuple[List[Dict], Dict[str, int], Dict[str, int], List[Dict], List[Dict]]:
    """
    Process a single log file.

    Args:
        args: Tuple of (log_file_path, linear_bounds, angular_bounds)

    Returns:
        Tuple of (claps_data, conversion_issues, quality_issues, workspace_rejected_segments, raw_segments)
    """
    log_file, linear_bounds, angular_bounds = args

    # Initialize results for this log file
    claps_data: List[Dict] = []
    conversion_issues: Dict[str, int] = defaultdict(int)
    quality_issues: Dict[str, int] = defaultdict(int)
    workspace_rejected_segments: List[Dict] = []
    raw_segments: List[Dict] = []  # Collect all raw segments for velocity analysis

    try:
        # Read log data for this file only
        log_data = read_lcm_experiment_log(log_file)
        if not log_data:
            conversion_issues["failed_log_read"] += 1
            return (
                claps_data,
                conversion_issues,
                quality_issues,
                workspace_rejected_segments,
                raw_segments,
            )

        # Extract segments for this file only
        segments = extract_motion_segments(log_data)

        # Load Vicon data for this file only
        vicon_data = find_vicon_state_data(log_file)

        # Process each segment
        for segment in segments:
            # Collect raw segment for velocity analysis
            raw_segments.append(segment)

            # Validate segment quality
            is_valid, issues = validate_segment_quality(segment)

            if not is_valid:
                for issue in issues:
                    quality_issues[issue] += 1
                continue

            # Extract poses using only this file's Vicon data
            start_pose, end_pose = extract_segment_poses(vicon_data, segment)

            if start_pose is None:
                conversion_issues["missing_start_pose"] += 1
                continue

            if end_pose is None:
                conversion_issues["missing_end_pose"] += 1
                continue

            # Check workspace bounds
            start_in_workspace = is_pose_in_safe_workspace(start_pose[0], start_pose[1])
            end_in_workspace = is_pose_in_safe_workspace(end_pose[0], end_pose[1])

            if not start_in_workspace or not end_in_workspace:
                conversion_issues["outside_workspace"] += 1
                rejected_segment = {
                    "segment_name": segment["test_name"],
                    "start_pose": start_pose,
                    "end_pose": end_pose,
                    "reason": f"start_outside={not start_in_workspace}, end_outside={not end_in_workspace}",
                }
                workspace_rejected_segments.append(rejected_segment)
                continue

            # Extract velocities
            measured_twists = segment["measured_twists"]
            if len(measured_twists) < 2:
                conversion_issues["insufficient_twist_data"] += 1
                continue

            first_twist = measured_twists[0]
            last_twist = measured_twists[-1]

            # Filter negative body-frame forward velocity
            vx_body_start = first_twist[1]
            vx_body_end = last_twist[1]
            if vx_body_start < 0 or vx_body_end < 0:
                conversion_issues["negative_body_frame_vx"] += 1
                continue

            # Check velocity bounds
            wz_start = first_twist[3]
            wz_end = last_twist[3]

            if (
                vx_body_start < linear_bounds[0]
                or vx_body_start > linear_bounds[1]
                or vx_body_end < linear_bounds[0]
                or vx_body_end > linear_bounds[1]
            ):
                conversion_issues["outside_linear_velocity_bounds"] += 1
                continue

            if abs(wz_start) > angular_bounds[1] or abs(wz_end) > angular_bounds[1]:
                conversion_issues["exceeds_angular_velocity_bound"] += 1
                continue

            # Process segment to CLAPS format
            claps_segment = process_segment_to_claps(
                segment, start_pose, end_pose, first_twist, last_twist
            )
            if claps_segment:
                claps_data.append(claps_segment)

    except Exception as e:
        conversion_issues[f"processing_error_{type(e).__name__}"] += 1
        error_msg = f"Error processing {log_file}: {e}"
        print(error_msg)
        # Also log with traceback if logger is available in the future
        import traceback

        print(f"Full traceback: {traceback.format_exc()}")

    return (
        claps_data,
        conversion_issues,
        quality_issues,
        workspace_rejected_segments,
        raw_segments,
    )


def process_segment_to_claps(
    segment: Dict,
    start_pose: Tuple,
    end_pose: Tuple,
    first_twist: List,
    last_twist: List,
) -> Optional[Dict]:
    """
    Convert a single segment to CLAPS format.

    Args:
        segment: Motion segment dictionary
        start_pose: (x, y, theta) start pose
        end_pose: (x, y, theta) end pose
        first_twist: [t, vx, vy, wz] at start
        last_twist: [t, vx, vy, wz] at end

    Returns:
        CLAPS format dictionary or None if conversion fails
    """
    try:
        # Wrap angles
        start_yaw_wrapped = math.atan2(math.sin(start_pose[2]), math.cos(start_pose[2]))
        end_yaw_wrapped = math.atan2(math.sin(end_pose[2]), math.cos(end_pose[2]))

        # Transform velocities
        vx0_world, vy0_world, wz0_world = body_to_world_velocity(
            first_twist[1], first_twist[2], first_twist[3], start_yaw_wrapped
        )
        vx1_world, vy1_world, wz1_world = body_to_world_velocity(
            last_twist[1], last_twist[2], last_twist[3], end_yaw_wrapped
        )

        # Compute angle difference
        angle_difference = math.atan2(
            math.sin(end_yaw_wrapped - start_yaw_wrapped),
            math.cos(end_yaw_wrapped - start_yaw_wrapped),
        )

        # Rotate coordinate system
        rotation_angle = -start_yaw_wrapped
        cos_rot = math.cos(rotation_angle)
        sin_rot = math.sin(rotation_angle)

        vx0_rotated = vx0_world * cos_rot - vy0_world * sin_rot
        vy0_rotated = vx0_world * sin_rot + vy0_world * cos_rot
        vx1_rotated = vx1_world * cos_rot - vy1_world * sin_rot
        vy1_rotated = vx1_world * sin_rot + vy1_world * cos_rot

        # Rotate positions
        x0_rotated = start_pose[0] * cos_rot - start_pose[1] * sin_rot
        y0_rotated = start_pose[0] * sin_rot + start_pose[1] * cos_rot
        x1_rotated = end_pose[0] * cos_rot - end_pose[1] * sin_rot
        y1_rotated = end_pose[0] * sin_rot + end_pose[1] * cos_rot

        # Translate to origin
        x0_final = 0.0
        y0_final = 0.0
        x1_final = x1_rotated - x0_rotated
        y1_final = y1_rotated - y0_rotated

        start_yaw_rotated = 0.0
        end_yaw_rotated = angle_difference

        # Create CLAPS format data
        claps_segment = {
            "q0": torch.tensor(
                [x0_final, y0_final, start_yaw_rotated], dtype=torch.float64
            ),
            "q1": torch.tensor(
                [x1_final, y1_final, end_yaw_rotated], dtype=torch.float64
            ),
            "dq0": torch.tensor(
                [vx0_rotated, vy0_rotated, wz0_world], dtype=torch.float64
            ),
            "dq1": torch.tensor(
                [vx1_rotated, vy1_rotated, wz1_world], dtype=torch.float64
            ),
            "u": torch.tensor(
                [segment["command"]["ax"], segment["command"]["az"]],
                dtype=torch.float64,
            ),
            "segment_name": segment["test_name"],
            "duration": segment["actual_duration"],
            "original_q0": start_pose,
            "original_q1": end_pose,
        }

        return claps_segment
    except Exception as e:
        error_msg = f"Error converting segment {segment.get('test_name', 'unknown')} to CLAPS format: {e}"
        print(error_msg)
        # Add more detailed logging for debugging
        import traceback

        print(f"Segment data: start_pose={start_pose}, end_pose={end_pose}")
        print(f"Twist data: first_twist={first_twist}, last_twist={last_twist}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None


def body_to_world_velocity(vx_body, vy_body, wz, yaw):
    """Convert body-frame velocity [vx_body, vy_body, wz] to world-frame [vx_world, vy_world, wz]"""
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    vx_world = vx_body * cos_yaw - vy_body * sin_yaw
    vy_world = vx_body * sin_yaw + vy_body * cos_yaw
    return vx_world, vy_world, wz


def process_logs_to_claps_format(
    log_files: List[str],
    logger: Optional[logging.Logger] = None,
    use_parallel: bool = True,
) -> Tuple[List[Dict], Dict[str, Any], List[Dict], List[Dict]]:
    """
    Returns:
        (claps_data, conversion_report, workspace_rejected_segments, raw_segments): CLAPS format data, detailed conversion report, workspace rejected segments, and raw segments for analysis
    """
    # Load velocity bounds from systems.yaml
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "systems.yaml"
    )
    with open(config_file, "r") as f:
        systems_config = yaml.load(f, Loader=yaml.FullLoader)

    real_mbot_config = systems_config["Real_MBot"]
    vel_bounds = real_mbot_config["vel_bounds"]
    linear_bounds = vel_bounds[0]  # [min_vx, max_vx]
    angular_bounds = vel_bounds[1]  # [min_wz, max_wz]

    print(f"🎯 Loaded velocity bounds from systems.yaml:")
    print(f"   Linear velocity (vx): [{linear_bounds[0]}, {linear_bounds[1]}] m/s")
    print(f"   Angular velocity (wz): [{angular_bounds[0]}, {angular_bounds[1]}] rad/s")

    # Initialize conversion report
    conversion_report: Dict[str, Any] = {
        "input_log_files": len(log_files),
        "successful_conversions": 0,
        "conversion_issues": defaultdict(int),
        "quality_issues": defaultdict(int),
    }

    claps_data = []
    workspace_rejected_segments = []
    raw_segments = [] 

    run_parallel = False
    file_results = None
    parallel_failure_reason = None

    if run_parallel:
        if logger:
            logger.info(f"🚀 Processing {len(log_files)} log files in parallel...")
        else:
            print(f"🚀 Processing {len(log_files)} log files in parallel...")

        # Limit parallelism based on number of files and CPU cores
        # Cap at 8 workers to prevent "too many open files" errors
        num_workers = min(8, mp.cpu_count(), len(log_files))

        if logger:
            logger.info(f"   Using {num_workers} workers")

        # Prepare arguments for parallel processing - each worker gets one log file
        file_args = [
            (log_file, linear_bounds, angular_bounds) for log_file in log_files
        ]

        try:
            with mp.Pool(processes=num_workers, maxtasksperchild=1) as pool:
                # Use chunksize=1 to avoid holding many file descriptors at once
                file_results = pool.map(process_log_file, file_args, chunksize=1)
        except Exception as exc:
            parallel_failure_reason = f"{type(exc).__name__}: {exc}"
            warning_msg = (
                "⚠️ Parallel processing failed and will retry sequentially. "
                f"Reason: {parallel_failure_reason}. "
                "Re-run with --no-parallel to skip multiprocessing."
            )
            if logger:
                logger.warning(warning_msg)
            else:
                print(warning_msg)
            run_parallel = False

    if run_parallel and file_results is not None:
        # Aggregate results from all files
        for (
            file_claps_data,
            file_conversion_issues,
            file_quality_issues,
            file_workspace_rejected,
            file_raw_segments,
        ) in file_results:
            claps_data.extend(file_claps_data)
            workspace_rejected_segments.extend(file_workspace_rejected)
            raw_segments.extend(file_raw_segments)

            # Aggregate issues
            for issue, count in file_conversion_issues.items():
                conversion_report["conversion_issues"][issue] += count
            for issue, count in file_quality_issues.items():
                conversion_report["quality_issues"][issue] += count

        conversion_report["successful_conversions"] = len(claps_data)

        if logger:
            logger.info(
                f"✅ Parallel processing complete: {len(claps_data)} valid segments"
            )
        else:
            print(f"✅ Parallel processing complete: {len(claps_data)} valid segments")

    else:
        # Sequential processing
        sequential_msg = f"🔄 Processing {len(log_files)} log files sequentially..."
        if parallel_failure_reason:
            sequential_msg += " (fallback from parallel failure)"

        if logger:
            logger.info(sequential_msg)
        else:
            print(sequential_msg)

        for log_file in log_files:
            # Process each file individually
            (
                file_claps_data,
                file_conversion_issues,
                file_quality_issues,
                file_workspace_rejected,
                file_raw_segments,
            ) = process_log_file((log_file, linear_bounds, angular_bounds))

            # Aggregate results
            claps_data.extend(file_claps_data)
            workspace_rejected_segments.extend(file_workspace_rejected)
            raw_segments.extend(file_raw_segments)

            for issue, count in file_conversion_issues.items():
                conversion_report["conversion_issues"][issue] += count
            for issue, count in file_quality_issues.items():
                conversion_report["quality_issues"][issue] += count

        conversion_report["successful_conversions"] = len(claps_data)

    return claps_data, conversion_report, workspace_rejected_segments, raw_segments



def visualize_data(
    claps_data: List[Dict],
    output_dir: str,
    workspace_rejected_segments: Optional[List[Dict]] = None,
):
    """
    Create 2D plots showing both workspace filtering (original coords) and normalized trajectories.

    Args:
        claps_data: List of valid CLAPS format data (in normalized coordinates)
        output_dir: Output directory for saving plots
        workspace_rejected_segments: List of segments rejected due to workspace bounds (optional, in original coords)
    """
    if not claps_data:
        print("No data to visualize")
        return

    # Extract normalized coordinates from CLAPS data (rotated + translated)
    x0_norm = [data["q0"][0].item() for data in claps_data]  # All should be 0.0
    y0_norm = [data["q0"][1].item() for data in claps_data]  # All should be 0.0
    x1_norm = [data["q1"][0].item() for data in claps_data]  # Relative motion
    y1_norm = [data["q1"][1].item() for data in claps_data]  # Relative motion

    # Create plot with workspace bounds (original coords) and normalized trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Workspace bounds with rejected segments (original Vicon coordinates)
    ax1.set_title("Workspace Filtering (Original Vicon Coordinates)")

    # Extract original coordinates from accepted segments
    accepted_x0 = [data["original_q0"][0] for data in claps_data]
    accepted_y0 = [data["original_q0"][1] for data in claps_data]
    accepted_x1 = [data["original_q1"][0] for data in claps_data]
    accepted_y1 = [data["original_q1"][1] for data in claps_data]

    # Plot accepted trajectories in original coordinates
    ax1.scatter(
        accepted_x0,
        accepted_y0,
        c="green",
        alpha=0.6,
        s=20,
        label="Valid start positions",
    )
    ax1.scatter(
        accepted_x1, accepted_y1, c="red", alpha=0.6, s=20, label="Valid end positions"
    )

    # Plot rejected poses if available (these are in original coordinates)
    if workspace_rejected_segments:
        rejected_x0 = []
        rejected_y0 = []
        rejected_x1 = []
        rejected_y1 = []

        for rejected_segment in workspace_rejected_segments:
            start_pose = rejected_segment["start_pose"]
            end_pose = rejected_segment["end_pose"]
            rejected_x0.append(start_pose[0])
            rejected_y0.append(start_pose[1])
            rejected_x1.append(end_pose[0])
            rejected_y1.append(end_pose[1])

        if rejected_x0:
            ax1.scatter(
                rejected_x0,
                rejected_y0,
                c="gray",
                alpha=0.6,
                s=20,
                label="Rejected start positions",
                marker="x",
            )
            ax1.scatter(
                rejected_x1,
                rejected_y1,
                c="gray",
                alpha=0.6,
                s=20,
                label="Rejected end positions",
                marker="x",
            )
            print(f"📊 Plotted {len(rejected_x0)} workspace-rejected pose pairs")

    # Draw workspace boundaries (in original coordinates)
    from matplotlib.patches import Rectangle

    workspace_rect = Rectangle(
        (SAFE_WORKSPACE_BOUNDS["x_min"], SAFE_WORKSPACE_BOUNDS["y_min"]),
        SAFE_WORKSPACE_BOUNDS["x_max"] - SAFE_WORKSPACE_BOUNDS["x_min"],
        SAFE_WORKSPACE_BOUNDS["y_max"] - SAFE_WORKSPACE_BOUNDS["y_min"],
        linewidth=2,
        edgecolor="blue",
        facecolor="none",
        linestyle="--",
        alpha=0.8,
        label="Safe workspace bounds",
    )
    ax1.add_patch(workspace_rect)
    ax1.set_xlabel("X (m) - Original Vicon Frame")
    ax1.set_ylabel("Y (m) - Original Vicon Frame")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # Right plot: Normalized trajectories (all start at origin)
    ax2.set_title("Normalized Trajectories (Rotated & Translated)")
    ax2.scatter(
        x0_norm,
        y0_norm,
        c="green",
        alpha=0.6,
        s=20,
        label="Start positions (all at origin)",
    )
    ax2.scatter(
        x1_norm,
        y1_norm,
        c="red",
        alpha=0.6,
        s=20,
        label="End positions (relative motion)",
    )

    ax2.set_xlabel("ΔX (m) - Relative Motion")
    ax2.set_ylabel("ΔY (m) - Relative Motion")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis("equal")

    plt.tight_layout()
    # Save plot
    plot_path = os.path.join(output_dir, "reports", "real_mbot_data_visualization.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # Explicitly close the figure to free memory

    print(f"📊 Visualization saved to: {plot_path}")
    print(f"📊 Left plot: Workspace filtering in original Vicon coordinates")
    print(
        f"📊 Right plot: {len(claps_data)} normalized trajectories (all start at origin)"
    )


def split_and_save_data(claps_data: List[Dict], split_ratio: float, output_dir: str):
    """
    Shuffle data, split into calibration/validation, and save as .pt files.

    Args:
        claps_data: List of CLAPS-format data
        split_ratio: Fraction for calibration (e.g., 0.1 for 10%)
        output_dir: Output directory (data/Real_MBot/)
    """
    if not claps_data:
        print("❌ No data to split and save")
        return

    # Create output directories and clean existing data
    cal_dir = os.path.join(output_dir, "raw_data", "calibration")
    val_dir = os.path.join(output_dir, "raw_data", "validation")

    # Clean existing files to avoid stale data
    import shutil

    if os.path.exists(cal_dir):
        shutil.rmtree(cal_dir)
        print(f"🧹 Cleaned existing calibration data: {cal_dir}")
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
        print(f"🧹 Cleaned existing validation data: {val_dir}")

    os.makedirs(cal_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Shuffle data to randomize time-series order
    shuffled_data = claps_data.copy()
    random.shuffle(shuffled_data)

    # Split data
    n_cal = int(len(shuffled_data) * split_ratio)
    cal_data = shuffled_data[:n_cal]
    val_data = shuffled_data[n_cal:]

    print(f"📊 Data split: {len(cal_data)} calibration, {len(val_data)} validation")

    # Save calibration data (one file per segment)
    for i, data in enumerate(cal_data):
        cal_dict = {
            "q0": data["q0"].unsqueeze(
                0
            ),  # Add batch dimension for CLAPS compatibility
            "q1": data["q1"].unsqueeze(0),
            "dq0": data["dq0"].unsqueeze(0),
            "dq1": data["dq1"].unsqueeze(0),
            "u": data["u"].unsqueeze(0),
        }

        cal_file = os.path.join(cal_dir, f"cal_real_robot_{i:04d}.pt")
        torch.save(cal_dict, cal_file)

    # Save validation data (one file per segment)
    for i, data in enumerate(val_data):
        val_dict = {
            "q0": data["q0"].unsqueeze(0),  # Add batch dimension
            "q1": data["q1"].unsqueeze(0),
            "dq0": data["dq0"].unsqueeze(0),
            "dq1": data["dq1"].unsqueeze(0),
            "u": data["u"].unsqueeze(0),
        }

        val_file = os.path.join(val_dir, f"val_real_robot_{i:04d}.pt")
        torch.save(val_dict, val_file)

    print(f"✅ Saved {len(cal_data)} calibration files to {cal_dir}")
    print(f"✅ Saved {len(val_data)} validation files to {val_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Process Real MBot LCM data for CLAPS pipeline"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--log_files", nargs="+", help="Paths to LCM log files")
    group.add_argument("--log_dir", help="Directory containing LCM log files")
    parser.add_argument(
        "--output_dir", default="data/Real_MBot", help="Output directory for data files (logs stored in logs/real_mbot_processing/)"
    )
    parser.add_argument(
        "--split_ratio", type=float, help="Calibration split ratio (overrides config)"
    )
    parser.add_argument("--seed", type=int, default=21, help="Random seed for shuffling")
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.output_dir, verbose=args.verbose)
    logger.info("=== Starting Real MBot Data Processing ===")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Parallel processing: {'disabled' if args.no_parallel else 'enabled'}")

    # Set random seed
    random.seed(args.seed)

    # Get log files
    if args.log_files:
        log_files = args.log_files
    else:
        log_files = glob.glob(os.path.join(args.log_dir, "*.lcm"))

    if not log_files:
        print("❌ No log files found")
        return 1

    print(f"🔍 Found {len(log_files)} log files:")
    for log_file in log_files:
        print(f"  - {log_file}")

    # Load systems config for split ratio
    config_file = os.path.join(script_dir, "systems.yaml")
    with open(config_file, "r") as f:
        systems_config = yaml.load(f, Loader=yaml.FullLoader)

    real_mbot_config = systems_config.get("Real_MBot", {})
    if args.split_ratio is not None:
        split_ratio = args.split_ratio
    else:
        split_ratio = real_mbot_config.get("calibration", {}).get("split_ratio")

    print(f"📊 Using split ratio: {split_ratio} for calibration")

    # Process logs using existing exp_utils functions
    print(f"\n📖 Processing {len(log_files)} log files...")
    claps_data, conversion_report, workspace_rejected_segments, raw_segments = (
        process_logs_to_claps_format(
            log_files, logger=logger, use_parallel=not args.no_parallel
        )
    )

    # Analyze duration statistics after conversion (only for segments that passed quality checks)
    if claps_data:
        durations = [data["duration"] for data in claps_data]

        print(f"\n📊 DURATION STATISTICS (Final Data After Quality Filtering):")
        print(
            f"   Actual duration - Min: {min(durations):.4f}s, Max: {max(durations):.4f}s, Mean: {np.mean(durations):.4f}s ± {np.std(durations):.4f}s"
        )

        # Control rate analysis (25Hz = 0.04s period)
        control_period = 1.0 / 25.0  # 0.04s
        print(f"   Control period: {control_period:.4f}s (25Hz)")

        # Show duration distribution
        from collections import Counter

        duration_counts = Counter([round(d, 2) for d in durations])
        print(
            f"   Duration patterns: {dict(sorted(duration_counts.items())[:5])}"
        )  # Show top 5
    print(f"✅ Converted {len(claps_data)} segments to CLAPS format")

    # Print detailed rejection breakdown
    print(f"\n📊 DETAILED REJECTION BREAKDOWN:")
    print(f"   Total log files processed: {conversion_report['input_log_files']}")
    print(
        f"   Successfully converted segments: {conversion_report['successful_conversions']}"
    )

    # Quality issues (pre-conversion filtering)
    if conversion_report["quality_issues"]:
        print(f"\n   Quality Issues (filtered before conversion):")
        total_quality_rejected = sum(conversion_report["quality_issues"].values())
        for issue, count in sorted(
            conversion_report["quality_issues"].items(), key=lambda x: -x[1]
        ):
            print(f"      {issue}: {count}")
        print(f"      Total quality rejections: {total_quality_rejected}")

    # Conversion issues (during conversion)
    if conversion_report["conversion_issues"]:
        print(f"\n   Conversion Issues (failed during processing):")
        total_conversion_rejected = sum(conversion_report["conversion_issues"].values())
        for issue, count in sorted(
            conversion_report["conversion_issues"].items(), key=lambda x: -x[1]
        ):
            print(f"      {issue}: {count}")
        print(f"      Total conversion rejections: {total_conversion_rejected}")

    # Summary
    print(f"\n   SUMMARY:")
    print(
        f"      Final segments extracted: {conversion_report['successful_conversions']}"
    )
    if workspace_rejected_segments:
        print(f"      Workspace rejected: {len(workspace_rejected_segments)}")

    if not claps_data:
        print("❌ No valid data extracted")
        return 1

    # Sanity check: expect ~500 transitions per log file
    expected_per_file = 500
    actual_per_file = len(claps_data) / len(log_files)
    print(
        f"📊 Sanity check: {actual_per_file:.0f} transitions per file (expected ~{expected_per_file})"
    )

    if actual_per_file < expected_per_file * 0.5:  # Less than 50% expected
        print(
            f"⚠️ Warning: Low transition count. Expected ~{expected_per_file} per file."
        )

    # Load velocity bounds for velocity analysis
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "systems.yaml"
    )
    with open(config_file, "r") as f:
        systems_config = yaml.load(f, Loader=yaml.FullLoader)

    real_mbot_config = systems_config["Real_MBot"]
    vel_bounds = real_mbot_config["vel_bounds"]
    velocity_bounds = {
        "linear_bounds": vel_bounds[0],  # [min_vx, max_vx]
        "angular_bounds": vel_bounds[1],  # [min_wz, max_wz]
    }

    # Analyze velocity distributions before and after filtering
    print(f"\n📊 Analyzing velocity distributions...")
    analyze_velocity_distribution(
        raw_segments, claps_data, velocity_bounds, args.output_dir, logger
    )

    visualize_data(claps_data, args.output_dir, workspace_rejected_segments)

    # Split and save data
    split_and_save_data(claps_data, split_ratio, args.output_dir)

    print(f"\n✅ Processing complete! Data saved to {args.output_dir}")
    print(f"📁 Use --robot_type Real_MBot with full_run.py to run CLAPS pipeline")
    print(f"📋 Next steps:")
    print(
        f"   1. python scripts/full_run.py --robot_type Real_MBot --run-calibration --method CLAPS"
    )
    print(
        f"   2. python scripts/full_run.py --robot_type Real_MBot --run-validation-mesh --method CLAPS"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
