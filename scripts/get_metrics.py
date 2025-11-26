#!/usr/bin/env python3
import gc
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from concurrent.futures.process import BrokenProcessPool
except ImportError:
    BrokenProcessPool = RuntimeError
    BrokenProcessPool = RuntimeError
from multiprocessing import cpu_count, Pool

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
import yaml
from scipy.spatial import ConvexHull
from scipy.stats import chi2
from shapely.geometry import MultiPoint, Polygon
from shapely.plotting import plot_polygon
from tqdm import tqdm

import luis_utils.load

# Monkeypatch script_starter to avoid CUDA init side-effect when importing full_run
_original_script_starter = luis_utils.load.script_starter
luis_utils.load.script_starter = lambda *args, **kwargs: None

sys.path.append(str(Path(__file__).parent))
from full_run import methods as FULL_RUN_METHODS

# Restore original script_starter
luis_utils.load.script_starter = _original_script_starter

# Ensure double precision (was previously set by full_run import side-effect)
torch.set_default_dtype(torch.float64)


# --- Inlined from metrics_utils.py ---
def count_calibration_data(robot_name):
    """Count calibration data files for the given robot."""
    calibration_dir = Path(f"data/{robot_name}/raw_data/calibration/")
    if calibration_dir.exists():
        calibration_files = list(calibration_dir.glob("*.pt"))
        return len(calibration_files)
    return 0


def count_validation_data(robot_name):
    """Count validation data files for the given robot."""
    validation_dir = Path(f"data/{robot_name}/raw_data/validation/")
    if validation_dir.exists():
        validation_files = list(validation_dir.glob("*.pt"))
        return len(validation_files)
    return 0


def validation_count_mismatch(summary_path: Path, robot_name: str):
    """
    Compare recorded validation count in summary_statistics.pt with current files.

    Returns:
        (mismatch: bool, current_count: int, recorded_count: int)
    """
    recorded = None
    current = count_validation_data(robot_name)

    if summary_path.exists():
        try:
            stats = torch.load(summary_path, map_location="cpu", weights_only=False)
            recorded = stats.get("metadata", {}).get("n_validation_runs")
        except Exception:
            # If we cannot read the summary, force recompute
            return True, current, recorded

    if recorded is None:
        return True, current, recorded

    return current != recorded, current, recorded


def recompute_enabled(skip_recompute_flag: bool) -> bool:
    """Return True when recomputation should occur (default unless explicitly skipped)."""
    return not skip_recompute_flag


# --- Inlined from parallel_settings.py ---
def resolve_method_parallelism(mode, parallel_methods_flag):
    """Return True when method processing should run in parallel.

    Args:
        mode: Execution mode string such as 'paper-plots' or None.
        parallel_methods_flag: Tri-state CLI flag; True, False, or None when unset.
    """
    if parallel_methods_flag is not None:
        return parallel_methods_flag
    # Default to parallel for all modes now that we have stable multiprocessing
    return True


def choose_method_workers(requested_workers):
    """Return worker count for method processing with a conservative default."""
    available = cpu_count()
    if requested_workers is None:
        return min(available, 2)
    return max(1, min(requested_workers, available))


# Maximum workers for multiprocessing (Phase 1 polygons)
MAX_WORKERS = 4

# Consistent color scheme for methods across all plots
METHOD_COLORS = {
    "BASELINE1": "#ff8c00",  # Orange
    "BASELINE2": "#6f00ff",  # Purple
    "BASELINE3": "#ff00d4",  # Pink
    "BASELINE4": "#0011ff",  # Blue
    "BASELINE5": "#ff0000",  # Red
    "BASELINE6": "#03f8fc",  # Cyan
    "BASELINE7": "#808080",  # Gray
    "CLAPS": "#00FF00",  # Green
}

METHOD_DISPLAY_NAMES = {k: v["name"] for k, v in FULL_RUN_METHODS.items()}

PAPER_FIGURE_METHOD_ORDER = [
    "BASELINE5",
    "BASELINE4",
    "BASELINE1",
    "BASELINE6",
    "BASELINE7",
    "CLAPS",
]


METHOD_LINESTYLES = {
    "BASELINE2": ":",
    "BASELINE3": "--",
    "BASELINE6": "-.",
    "BASELINE7": "--",
}

# Draw Monte Carlo particle clouds behind contour outlines
MC_PARTICLES_ZORDER = 0

BASELINE7_TILDE_POINT_SIZE = 40

# DPI for all saved figures
FIGURE_DPI = 300


def get_boundary_fraction(method_name):
    """Return boundary sampling fraction used for polygon fitting."""

    upper_name = method_name.upper() if method_name else ""

    if "MC_PARTICLES" in upper_name:
        return 0.1

    if upper_name in {"BASELINE6", "BASELINE7"}:
        return 0.3

    return 0.3


def get_method_linestyle(method_name):
    """Return consistent linestyle for each method with a solid fallback."""

    return METHOD_LINESTYLES.get(method_name, "-")


def get_paper_figure_azimuth_angles(validation_id):
    """Return azimuth angles for the paper-style 3D plots."""

    if validation_id == "0301":
        return [315]

    return list(range(0, 360, 1))


def select_tilde_q1_point(method_map):
    """Pick a 2D mean point for plotting, preferring pre-bias values."""
    import numpy as np

    preferred_methods = ["BASELINE7", "BASELINE6", "CLAPS"]

    ordered_keys = preferred_methods + [k for k in method_map.keys() if k not in preferred_methods]
    for key in ordered_keys:
        data = method_map.get(key)
        if not data:
            continue
        coord = data.get("tilde_q1_pre_bias")
        if coord is None:
            coord = data.get("tilde_q1")
        if coord is None:
            continue

        coord_np = coord.cpu().numpy() if hasattr(coord, "cpu") else np.array(coord)
        if coord_np.ndim == 1:
            return coord_np[0], coord_np[1]
        return coord_np[0, 0], coord_np[0, 1]

    return None


def select_common_and_mle_tilde_points(method_map):
    """Return (common_mean, baseline7_mean) 2D points for plotting.

    common_mean comes from the first available method in a preferred order that
    should reflect the unshifted prediction. baseline7_mean comes from the
    bias-corrected BASELINE7 output so it can be plotted separately. Raises on
    invalid point shapes to avoid silent fallbacks.
    """

    def to_xy(point_data, context):
        if point_data is None:
            raise ValueError(f"{context} is required for mean plotting")

        point_np = (
            point_data.cpu().numpy() if hasattr(point_data, "cpu") else np.array(point_data)
        )

        if point_np.ndim == 1 and point_np.shape[0] >= 2:
            return point_np[0], point_np[1]

        if point_np.ndim >= 2 and point_np.shape[1] >= 2:
            return point_np[0, 0], point_np[0, 1]

        raise ValueError(f"{context} must provide at least two position values")

    preferred_common = [
        "CLAPS",
        "BASELINE1",
        "BASELINE4",
        "BASELINE5",
        "BASELINE2",
        "BASELINE3",
        "BASELINE6",
        "BASELINE7",
    ]

    common_point = None
    for name in preferred_common:
        method_data = method_map.get(name)
        if method_data is None:
            continue
        if "tilde_q1" in method_data:
            common_point = to_xy(method_data["tilde_q1"], f"{name} tilde_q1")
            break

    if common_point is None:
        raise ValueError("No method provided a tilde_q1 point for plotting")

    baseline7_point = None
    baseline7_data = method_map.get("BASELINE7")
    if baseline7_data is not None:
        if "tilde_q1" not in baseline7_data:
            raise ValueError("BASELINE7 must provide tilde_q1 for shifted mean plotting")
        baseline7_point = to_xy(baseline7_data["tilde_q1"], "BASELINE7 tilde_q1")

    return common_point, baseline7_point


def save_figure_all_formats(
    figure, base_path: Path, dpi, bbox_inches, formats=("png",)
):
    """Save a matplotlib figure to the requested formats using shared parameters."""
    save_kwargs = {}
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches

    base_path = base_path if base_path.suffix == "" else base_path.with_suffix("")
    for extension in formats:
        figure.savefig(base_path.with_suffix(f".{extension}"), **save_kwargs)


def plot_two_contours(
    id1,
    id2,
    robot_name,
    confidence_level,
    boundary_points,
    output_path=None,
):
    """Plot contours for two validation runs side by side; returns True on success."""

    method_map1, mc_particles1, _ = load_validation_data_for_id(
        id1, robot_name, confidence_level, boundary_points
    )
    method_map2, mc_particles2, _ = load_validation_data_for_id(
        id2, robot_name, confidence_level, boundary_points
    )

    if not method_map1 or not method_map2:
        return False

    # --- DEBUG PRINT START ---
    def print_scenario_info(method_map, val_id):
        if not method_map:
            return
        any_d = next(iter(method_map.values()))
        dq0 = any_d.get("dq0")
        u = any_d.get("u")
        print(f"\n🔍 Scenario Info for {val_id}:")
        if dq0 is not None:
            print(f"   dq0: {dq0.cpu().numpy().flatten() if hasattr(dq0, 'cpu') else dq0}")
        if u is not None:
            print(f"   u:   {u.cpu().numpy().flatten() if hasattr(u, 'cpu') else u}")
        print("-" * 40)

    print_scenario_info(method_map1, id1)
    print_scenario_info(method_map2, id2)
    # --- DEBUG PRINT END ---

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=FIGURE_DPI)
    setup_2d_plot_axis(ax1, title="")
    setup_2d_plot_axis(ax2, title="")

    buffered_methods = ["CLAPS", "BASELINE1", "BASELINE4", "BASELINE5"]
    points_ax1 = plot_method_contours(
        ax1, method_map1, buffered_methods, collect_points_for_limits=True
    )
    points_ax2 = plot_method_contours(
        ax2, method_map2, buffered_methods, collect_points_for_limits=True
    )

    def add_mc(ax, mc_particles, point_list):
        if mc_particles is None or mc_particles.shape[0] == 0:
            return
        mc_np = mc_particles.cpu().numpy()
        point_list.append(mc_np[:, :2])
        ax.scatter(
            mc_np[:, 0],
            mc_np[:, 1],
            s=0.1,
            alpha=0.1,
            # label="Monte Carlo Samples",
            color="k",
            zorder=MC_PARTICLES_ZORDER,
        )
        ax.scatter(
            mc_np[:, 0].mean(),
            mc_np[:, 1].mean(),
            s=60,
            alpha=1.0,
            facecolor="white",
            edgecolor="black",
            linewidth=1.4,
            zorder=160,
            marker="s",
            label="Monte Carlo Mean",
        )

    add_mc(ax1, mc_particles1, points_ax1)
    add_mc(ax2, mc_particles2, points_ax2)

    for method_map, ax in ((method_map1, ax1), (method_map2, ax2)):
        # Plot shared prediction point
        tilde_point = select_tilde_q1_point(method_map)
        if tilde_point is not None:
            pred_x, pred_y = tilde_point
            ax.scatter(
                pred_x,
                pred_y,
                s=45,
                alpha=1.0,
                color="#0011ff",
                zorder=170,
                marker="o",
                # label=r"$\tilde g_1$",
            )

        # Plot BASELINE7 bias-corrected prediction point
        try:
            _, baseline7_point = select_common_and_mle_tilde_points(method_map)
            if baseline7_point is not None:
                b7_x, b7_y = baseline7_point
                baseline7_color = METHOD_COLORS["BASELINE7"]
                ax.scatter(
                    b7_x,
                    b7_y,
                    s=BASELINE7_TILDE_POINT_SIZE,
                    alpha=1.0,
                    facecolor=baseline7_color,
                    # edgecolor="black",
                    zorder=200,
                    marker="o",
                    linewidths=2.0,
                    label="_nolegend_",
                )
        except Exception as e:
            # Log warning instead of silent failure
            import logging
            logging.getLogger("get_metrics").warning(f"Could not plot BASELINE7 point: {e}")

    apply_buffered_axis_limits([ax1], points_ax1)
    apply_buffered_axis_limits([ax2], points_ax2)

    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    combined = []
    combined_labels = []
    seen = set()
    for h, l in list(zip(handles, labels)) + list(zip(handles2, labels2)):
        if l not in seen:
            combined.append(h)
            combined_labels.append(l)
            seen.add(l)

    fig.legend(
        combined,
        combined_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=5,
        fontsize=13,
        markerscale=0.8,
        scatterpoints=1,
    )

    plt.subplots_adjust(top=0.75, wspace=0.15)

    if output_path:
        base_path = Path(output_path).with_suffix("")
    else:
        save_dir = Path(
            f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}/plots_2d"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        base_path = save_dir / f"contour_comparison_{id1}_vs_{id2}"

    plt.tight_layout()
    save_figure_all_formats(
        fig, base_path, 300, "tight", formats=("png", "pdf", "svg")
    )
    plt.close(fig)
    return True


def two_contour_comparison(
    robot_name, confidence_level, boundary_points, logger
):
    """Generate the fixed two-contour comparison figure used for the paper."""
    target_robot = "Isaac_Jetbot"
    target_confidence = "over_confident"
    target_boundary_points = 5000

    if (
        robot_name != target_robot
        or confidence_level != target_confidence
        or boundary_points != target_boundary_points
    ):
        logger.info(
            "Skipping contour comparison plot (only generated for Isaac_Jetbot, over_confident, 5000)."
        )
        return

    id_one = "val_isaac_0007"
    id_two = "val_isaac_0590"
    try:
        success = plot_two_contours(
            id_one,
            id_two,
            robot_name,
            confidence_level,
            boundary_points,
            None,
        )
        if success:
            logger.info(
                f"✅ Generated fixed contour comparison: {id_one} vs {id_two}"
            )
        else:  # pragma: no cover - depends on data presence
            logger.warning(
                f"⚠️ Contour comparison failed for {id_one} vs {id_two}"
            )
    except Exception as exc:  # pragma: no cover - depends on runtime data
        logger.error(f"❌ Error generating contour comparison plot: {exc}")


def setup_logging(verbose: bool = True) -> logging.Logger:
    """Set up comprehensive logging to file and console for get_metrics output."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)
    logs_dir = os.path.join(workspace_root, "logs", "metrics")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"get_metrics_{timestamp}.log")

    logger = logging.getLogger("get_metrics")
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

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


# Known timing metric keys
TIMING_METRICS = {
    "dynamics_predict_total",
    "error_total",
    "online_calibration_total",
    "sample_boundary_total",
    "mesh_fitting_total",
    "check_points_in_mesh_total",
    "cgal_delaunay_only",
    "pyvista_processing",
    "pyvista_delaunay",
    "calibration_total",
    "full_algorithm_total_s",
}


def format_tensor(tensor, precision=2):
    """Converts a tensor to a clean string format for plot titles."""
    if tensor is None:
        return "N/A"
    elements = [f"{x:.{precision}f}" for x in tensor.cpu().numpy()]
    return f"[{', '.join(elements)}]"


def format_tensor_for_filename(tensor, precision=2):
    """Converts a tensor to a filename-safe string format."""
    if tensor is None:
        return "NA"
    elements = [f"{x:.{precision}f}" for x in tensor.cpu().numpy()]
    print(elements)
    print(tensor)
    return "_".join(elements).replace(".", "p").replace("-", "m")


def extract_boundary_points(points_2d, boundary_fraction):
    """Extract boundary points using CV2 contour as distance filter.

    Args:
        points_2d: numpy array of shape (N, 2) with [x, y] coordinates
        boundary_fraction: fraction of points closest to contour boundary to keep

    Returns:
        numpy array of boundary points
    """
    import cv2
    from scipy.spatial import cKDTree

    if len(points_2d) < 10:
        return points_2d

    # Create binary image from points
    x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
    y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()

    # Add padding
    padding = 0.1 * max(x_max - x_min, y_max - y_min)
    x_min, x_max = x_min - padding, x_max + padding
    y_min, y_max = y_min - padding, y_max + padding

    # Create image
    img_size = 500
    binary_img = np.zeros((img_size, img_size), dtype=np.uint8)

    # Convert points to image coordinates
    x_scale = (img_size - 1) / (x_max - x_min)
    y_scale = (img_size - 1) / (y_max - y_min)

    img_points = np.column_stack(
        [(points_2d[:, 0] - x_min) * x_scale, (points_2d[:, 1] - y_min) * y_scale]
    ).astype(int)

    # Fill image
    for pt in img_points:
        if 0 <= pt[0] < img_size and 0 <= pt[1] < img_size:
            binary_img[pt[1], pt[0]] = 255

    # Apply morphological operations to connect nearby points
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_img = cv2.dilate(binary_img, kernel, iterations=2)
    binary_img = cv2.erode(binary_img, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        # Fallback to all points if contour detection fails
        return points_2d

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.reshape(-1, 2)

    # Convert contour back to real coordinates
    real_contour = np.column_stack(
        [contour_points[:, 0] / x_scale + x_min, contour_points[:, 1] / y_scale + y_min]
    )

    # Now find original points that are close to this contour
    tree = cKDTree(real_contour)

    # For each original point, find distance to nearest contour point
    distances, _ = tree.query(points_2d, k=1)

    # Keep points within threshold distance of contour
    threshold = np.percentile(distances, boundary_fraction * 100)
    boundary_mask = distances <= threshold

    return points_2d[boundary_mask]


def fit_polygon_to_points(points_2d, method_name="unknown", run_id=None):
    """Fit a polygon to 2D points using alphashape and optionally compute workspace footprint.

    Args:
        points_2d: numpy array or torch tensor of shape (N, 2) with [x, y] coordinates
        method_name: name of the method for debug plots
        run_id: run ID for debug plots

    Returns:
        polygon: Shapely Polygon object
        area: area of the polygon
        coords: numpy array of polygon coordinates for plotting
    """
    if hasattr(points_2d, "cpu"):  # Check if it's a PyTorch tensor
        points_np = points_2d.cpu().numpy()
    else:
        points_np = points_2d

    if len(points_np) < 3:
        return None, 0.0, None, {}

    original_count = len(points_np)

    # Methods that are known to be convex - use fast ConvexHull path
    convex_methods = ["BASELINE1", "BASELINE2", "BASELINE3", "BASELINE5"]
    method_upper = method_name.upper()
    use_convex_hull = any(m in method_upper for m in convex_methods)

    if use_convex_hull:
        print(
            f"    → {method_name} {run_id}: Using fast ConvexHull on {original_count} points..."
        )
        import time

        start_time = time.time()

        from shapely.geometry import Polygon

        hull = ConvexHull(points_np)
        hull_points = points_np[hull.vertices]

        # Create polygon from hull
        polygon = Polygon(hull_points)
        coords = np.vstack([hull_points, hull_points[0]])  # Close the polygon
        area = polygon.area

        approach_name = "ConvexHull"
        processing_points = len(hull_points)
        elapsed = time.time() - start_time
        print(
            f"      ✓ Created polygon with {processing_points} vertices in {elapsed:.3f}s"
        )

        # Store debug data for later plotting
        debug_data = {
            "original_points": points_np,
            "boundary_points": hull_points,
            "approach_name": approach_name,
        }

    else:
        # SLOWER PATH: Boundary extraction + alphashape for concave methods
        print(
            f"    → {method_name} {run_id}: Processing {original_count} points with CV2+AlphaShape..."
        )

        # Special debug for MC_PARTICLES
        if "MC" in method_name.upper():
            print(f"      🔍 MC_PARTICLES DEBUG:")
            print(f"         Method name: '{method_name}'")
            print(f"         Upper case: '{method_name.upper()}'")
            print(
                f"         MC_PARTICLES check: {'MC_PARTICLES' in method_name.upper()}"
            )
            print(f"         Point count: {original_count}")

        import time

        start_time = time.time()
        approach_name = "CV2+AlphaShape"

        # Store original points for debug plotting before boundary extraction
        original_points_for_debug = points_np.copy()

        # Extract boundary points
        print(f"      Extracting boundary points from {original_count} points...")
        extract_start = time.time()
        boundary_fraction = get_boundary_fraction(method_name)
        boundary_pts = extract_boundary_points(
            points_np, boundary_fraction=boundary_fraction
        )
        extract_time = time.time() - extract_start
        print(
            f"      ✓ Reduced to {len(boundary_pts)} boundary points in {extract_time:.3f}s"
        )

        points_np = boundary_pts

        # Use alphashape with hardcoded alpha for non-convex shapes
        print(
            f"      Creating alphashape polygon from {len(points_np)} boundary points..."
        )
        poly_start = time.time()

        # Import alphashape locally to avoid multiprocessing issues
        import alphashape

        points_np = np.ascontiguousarray(points_np)

        alpha = 100
        polygon = alphashape.alphashape(points_np, alpha)

        gc.collect()

        poly_time = time.time() - poly_start
        print(
            f"      ✓ Created alphashape polygon with α={alpha:.3f} in {poly_time:.3f}s"
        )

        # Get coordinates for plotting (ensure polygon is closed)
        if hasattr(polygon, "exterior"):
            coords = np.array(polygon.exterior.coords)
            area = polygon.area
        else:
            raise ValueError("Polygon has no exterior")

        total_time = time.time() - start_time
        print(f"      Total processing time: {total_time:.3f}s")

        # Save enhanced debug plot showing boundary extraction + alphashape fitting
        print(
            f"      DEBUG: Checking visualization - original_count={original_count}, debug_points_available={original_points_for_debug is not None}"
        )

        # Store debug data for later plotting
        debug_data = {
            "original_points": original_points_for_debug,
            "boundary_points": points_np,
            "approach_name": approach_name,
            "boundary_fraction": boundary_fraction,
        }

    return polygon, area, coords, debug_data


def create_iou_debug_plot(run_id, method_name, polygon, mc_polygon, iou):
    """Create debug visualization for IoU calculation.

    Args:
        run_id: ID of validation run
        method_name: Name of method (only debugs CLAPS)
        polygon: Method's fitted polygon
        mc_polygon: MC particles polygon
        iou: Computed IoU value
    """
    # Only debug CLAPS with low IoU
    if method_name != "CLAPS" or iou >= 0.6:
        return

    print(f"  🔍 IoU DEBUG {run_id}: Generating debug plot for IoU={iou:.3f}")

    # Compute intersection and union
    intersection = polygon.intersection(mc_polygon)
    union = polygon.union(mc_polygon)

    # Handle MultiPolygon results
    if hasattr(intersection, "geoms"):
        intersection_area = sum(p.area for p in intersection.geoms)
    else:
        intersection_area = intersection.area

    if hasattr(union, "geoms"):
        union_area = sum(p.area for p in union.geoms)
    else:
        union_area = union.area

    print(f"             CLAPS area={polygon.area:.4f}, MC area={mc_polygon.area:.4f}")
    print(f"             Intersection={intersection_area:.4f}, Union={union_area:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Both polygons overlaid
    ax = axes[0]
    plot_polygon(mc_polygon, ax=ax, color="blue", alpha=0.3, label="MC")
    plot_polygon(polygon, ax=ax, color="red", alpha=0.3, label="CLAPS")
    ax.set_title(f"Both Polygons - {run_id}")
    ax.legend()
    ax.set_aspect("equal")

    # Panel 2: Intersection (green)
    ax = axes[1]
    if hasattr(intersection, "geoms"):  # MultiPolygon
        for poly in intersection.geoms:
            if poly.area > 0:  # Skip empty geometries
                plot_polygon(poly, ax=ax, color="green", alpha=0.7)
    else:
        if intersection.area > 0:
            plot_polygon(intersection, ax=ax, color="green", alpha=0.7)
    ax.set_title(f"Intersection (area={intersection_area:.4f})")
    ax.set_aspect("equal")

    # Panel 3: Union (purple)
    ax = axes[2]
    if hasattr(union, "geoms"):  # MultiPolygon
        for poly in union.geoms:
            if poly.area > 0:  # Skip empty geometries
                plot_polygon(poly, ax=ax, color="purple", alpha=0.5)
    else:
        if union.area > 0:
            plot_polygon(union, ax=ax, color="purple", alpha=0.5)
    ax.set_title(f"Union (area={union_area:.4f})")
    ax.set_aspect("equal")

    plt.suptitle(f"IoU Debug: {run_id} - IoU={iou:.3f}", fontsize=14)
    plt.tight_layout()

    # Save to debug folder
    debug_dir = Path("logs/debug/iou_debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(debug_dir / f"iou_{run_id}_{method_name}.png", dpi=FIGURE_DPI)
    plt.close(fig)
    plt.close("all")


def plot_polygon_fitting_debug(
    method_name,
    run_id,
    original_points,
    boundary_points,
    polygon,
    area,
    coords,
    approach_name,
):
    """Create debug visualization for polygon fitting process.

    Args:
        method_name: Method name for filename
        run_id: Run ID for filename
        original_points: Original point cloud
        boundary_points: Extracted boundary points (or hull points for convex)
        polygon: Fitted polygon
        area: Polygon area
        coords: Polygon coordinates
        approach_name: Algorithm approach name (ConvexHull or CV2+AlphaShape)
    """

    debug_dir = Path("logs/debug/boundary_extraction")
    debug_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%H%M%S_%f")[:9]
    if run_id is not None:
        run_id_str = str(run_id)
        file_stem = f"{method_name}_{run_id_str}_{timestamp}"
    else:
        file_stem = f"{method_name}_{timestamp}"

    figure_base_path = debug_dir / file_stem

    boundary_fraction_used = None
    if original_points is not None and len(original_points) > 0:
        boundary_fraction_used = len(boundary_points) / len(original_points)

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Left plot: original points
    ax1.scatter(
        original_points[:, 0], original_points[:, 1], alpha=0.05, s=0.5, c="blue"
    )
    ax1.set_title(f"Original: {len(original_points)} points")
    ax1.set_aspect("equal", adjustable="box")

    if approach_name == "ConvexHull":
        # Middle plot: Pre-buffer polygon (robot center path)
        ax2.scatter(
            original_points[:, 0],
            original_points[:, 1],
            alpha=0.1,
            s=1,
            c="gray",
            label="All points",
        )
        pre_buffer_coords = np.vstack([boundary_points, boundary_points[0]])
        ax2.plot(
            pre_buffer_coords[:, 0],
            pre_buffer_coords[:, 1],
            "r-",
            linewidth=2,
            label="Robot Center Path",
        )
        ax2.fill(pre_buffer_coords[:, 0], pre_buffer_coords[:, 1], "red", alpha=0.1)
        ax2.set_title(f"Pre-buffer: {len(boundary_points)} vertices")
        ax2.set_aspect("equal", adjustable="box")
        ax2.legend()
    else:
        # Middle plot: boundary extraction
        ax2.scatter(
            original_points[:, 0],
            original_points[:, 1],
            alpha=0.02,
            s=0.5,
            c="gray",
            label="All points",
        )
        ax2.scatter(
            boundary_points[:, 0],
            boundary_points[:, 1],
            s=5,
            c="red",
            alpha=0.8,
            label="Boundary",
        )
        ax2.set_title(f"Boundary: {len(boundary_points)} points")
        ax2.set_aspect("equal", adjustable="box")
        ax2.legend()

    # Right plot: polygon from boundary points
    ax3.scatter(
        boundary_points[:, 0],
        boundary_points[:, 1],
        s=3,
        c="red",
        alpha=0.6,
        label="Boundary points",
    )
    if polygon is not None and hasattr(polygon, "exterior"):
        ax3.plot(coords[:, 0], coords[:, 1], "g-", linewidth=2, label="Polygon")
        ax3.fill(coords[:, 0], coords[:, 1], "green", alpha=0.1)
        ax3.set_title(f"Polygon: Area={area:.4f}")
    else:
        ax3.set_title(f"Polygon: Failed")
    ax3.set_aspect("equal", adjustable="box")
    ax3.legend()

    if boundary_fraction_used is not None:
        plt.suptitle(
            f"Method: {approach_name} (kept {boundary_fraction_used:.2f} of points)"
        )
    else:
        plt.suptitle(f"Method: {approach_name}")
    plt.tight_layout()
    plt.savefig(figure_base_path.with_suffix('.png'), dpi=FIGURE_DPI)
    plt.close(fig)
    plt.close("all")


def filter_data_by_angle(data, reference_angle, angle_threshold):
    """Filter 3D data points to only include those with angle close to reference_angle.

    Args:
        data: numpy array or torch tensor of shape (N, 3) with [x, y, theta] coordinates
        reference_angle: target angle to filter around (in radians)
        angle_threshold: maximum absolute angle deviation from reference_angle (in radians)

    Returns:
        filtered_data: same type as input with only points where |theta - reference_angle| < angle_threshold
        mask: boolean mask indicating which points were kept
    """
    if data is None:
        return None, None

    if (hasattr(data, "ndim") and data.ndim < 2) or (
        hasattr(data, "dim") and data.dim() < 2
    ):
        # 1D tensor/array - return as is with full mask
        if isinstance(data, torch.Tensor):
            return data, torch.ones(data.shape[0], dtype=torch.bool)
        else:
            return data, np.ones(data.shape[0], dtype=bool)

    if data.shape[1] < 3:
        # 2D but less than 3 columns - return as is with full mask
        return data, torch.ones(data.shape[0], dtype=torch.bool)

    # Get the angle column (third dimension)
    angles = data[:, 2]

    # Calculate angular difference, handling wraparound
    angle_diff = np.abs(angles - reference_angle)
    # Handle wraparound: if difference > π, use 2π - difference
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    mask = angle_diff < angle_threshold

    filtered_data = data[mask]
    # Ensure filtered_data maintains 2D shape even with single points
    if filtered_data.ndim == 1:
        filtered_data = filtered_data.unsqueeze(0)
    return filtered_data, mask


def _compute_method_summary(metrics_list):
    """Compute summary statistics for a single method's metrics."""
    # Define skip keys for summary calculation
    SKIP_KEYS = {
        "boundary_SS_pts",
        "mesh_SS_pts",
        "mesh_SS_faces",
        "mesh_vertices",
        "mesh_faces",
        "q1",
        "q0",
        "tilde_q1",
        "polygon_coords",  # Variable-sized polygon coordinate arrays
        "polygon_2d",  # Shapely polygon object (not a tensor)
    }

    # Helper function for statistics
    def mean_std(dicts, key):
        vals = []
        for d in dicts:
            if key in d:
                v = d[key]
                if isinstance(v, (dict, list, str)):
                    continue
                if isinstance(v, bool):
                    v = torch.tensor(float(v))
                elif not isinstance(v, torch.Tensor):
                    v = torch.as_tensor(v)
                vals.append(v.cpu().float())
        if len(vals) == 0:
            return None
        stack = torch.stack(vals).cpu()
        return stack.mean().item(), stack.std(unbiased=False).item(), len(vals)

    # Helper function for timing statistics (handles both old and new formats)
    def timing_mean_std(dicts, key):
        all_times = []
        for d in dicts:
            if key in d:
                v = d[key]
                if isinstance(v, list):
                    # New format: list of timing values
                    all_times.extend([float(t) for t in v])
                elif isinstance(v, dict) and "mean_s" in v:
                    # Old format: dict with Welford statistics
                    # Use the mean value, but we lose the original data
                    all_times.append(float(v["mean_s"]))
                elif isinstance(v, (int, float)):
                    # Single timing value (like full_algorithm_total_s)
                    all_times.append(float(v))

        if len(all_times) == 0:
            return None

        return (
            np.mean(all_times),
            np.std(all_times, ddof=1) if len(all_times) > 1 else 0.0,
            len(all_times),
        )

    # Compute summary statistics
    keys = set().union(*(d.keys() for d in metrics_list)) - SKIP_KEYS
    summary_stats = {}

    for k in sorted(keys):
        if k in TIMING_METRICS:
            # Use specialized timing statistics
            tms = timing_mean_std(metrics_list, k)
            if tms is not None:
                mu, sd, n = tms
                summary_stats[k] = (mu, sd, n)
        else:
            # Use regular statistics
            msd = mean_std(metrics_list, k)
            if msd is not None:
                mu, sd, n = msd
                summary_stats[k] = (mu, sd, n)

    # Extract means, stds, and counts
    means = {k: v[0] for k, v in summary_stats.items()}
    stds = {k: v[1] for k, v in summary_stats.items()}
    counts = {k: v[2] for k, v in summary_stats.items()}

    # Store raw values for detailed analysis
    raw_values = {}
    for key in [
        "empirical_coverage_SS",
        "total_volume_mesh",
        "area_2d",
        "iou_vs_mc",
    ]:  # Key metrics including new 2D metrics
        values = []
        for d in metrics_list:
            if key in d:
                v = d[key]
                if isinstance(v, bool):
                    v = float(v)
                elif hasattr(v, "item"):
                    v = v.item()
                elif not isinstance(v, (int, float)):
                    continue
                values.append(v)
        if values:
            raw_values[key] = values

    return {
        "mean": means,
        "std": stds,
        "count": max(counts.values()) if counts else 0,
        "raw_values": raw_values,
    }


def convert_latex_to_pdf(latex_content, output_path, logger=None):
    """Convert LaTeX content to PDF using multiple LaTeX engines."""
    if logger is None:
        logger = logging.getLogger("get_metrics")

    # Create a temporary directory for LaTeX compilation
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Create a complete, self-contained LaTeX document
        full_doc = (
            r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{siunitx}
\usepackage{setspace}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{caption}
\usepackage{geometry}
\usepackage{graphicx}

% Define custom commands used in the table
\newcommand{\CPpredRegionQ}{C_q}
\newcommand{\methodName}{CLAPS}
\newcommand{\targetalpha}{0.10}

\geometry{margin=1in}

\begin{document}
"""
            + latex_content
            + r"""
\end{document}"""
        )

        full_tex_file = temp_dir / "full_doc.tex"
        with open(full_tex_file, "w") as f:
            f.write(full_doc)

        # Try different LaTeX engines
        latex_engines = ["lualatex", "xelatex", "pdflatex"]

        for engine in latex_engines:
            try:
                logger.info(f"Trying {engine}...")
                result = subprocess.run(
                    [
                        engine,
                        "-interaction=nonstopmode",
                        "-output-directory",
                        str(temp_dir),
                        str(full_tex_file),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    # Check for PDF output
                    pdf_file = temp_dir / "full_doc.pdf"
                    if pdf_file.exists():
                        import shutil

                        shutil.copy2(pdf_file, output_path)
                        logger.info(
                            f"✅ PDF generated successfully using {engine}: {output_path}"
                        )
                        return True
                    else:
                        logger.warning(f"⚠️  {engine} succeeded but no PDF found")
                        continue
                else:
                    logger.warning(
                        f"⚠️  {engine} failed with return code {result.returncode}"
                    )
                    # Show full error details to debug the issue
                    if result.stderr:
                        logger.debug(f"   STDERR:\n{result.stderr}")
                    if result.stdout:
                        logger.debug(f"   STDOUT:\n{result.stdout}")

                    # Also check for .log file which contains detailed LaTeX errors
                    log_file = temp_dir / "full_doc.log"
                    if log_file.exists():
                        with open(log_file, "r") as f:
                            log_content = f.read()
                        logger.debug(
                            f"   LaTeX LOG (last 1000 chars):\n{log_content[-1000:]}"
                        )

                    continue

            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️  {engine} timed out")
                continue
            except FileNotFoundError:
                logger.warning(f"⚠️  {engine} not found")
                continue

        logger.error(
            "❌ All LaTeX engines failed. You can manually compile the .tex file."
        )
        logger.error(f"   LaTeX file saved at: {output_path.with_suffix('.tex')}")
        return False


def load_scaling_factors(robot_name, confidence_level, boundary_points):
    """Load scaling factors for all methods from calibration files."""
    scaling_factors = {}

    # Get the experiment directory
    experiment_dir = Path(
        f"data/{robot_name}/experiments/{confidence_level}/boundary_{boundary_points}"
    )

    for method_name in FULL_RUN_METHODS.keys():
        method_cfg = FULL_RUN_METHODS[method_name]

        # Check if method needs calibration
        if not method_cfg.get("calibrate", False):
            scaling_factors[method_name] = "N/A (uncalibrated)"
        else:
            # Try to load scaling factor
            scaling_factor_path = (
                experiment_dir / method_name / "calibration" / "scaling_factor.pt"
            )
            try:
                if scaling_factor_path.exists():
                    data = torch.load(
                        scaling_factor_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    scaling_factors[method_name] = f"{data['scaling_factor']:.6f}"
                else:
                    scaling_factors[method_name] = "N/A (file not found)"
            except Exception as e:
                scaling_factors[method_name] = f"Error loading: {str(e)[:50]}"

    return scaling_factors


def print_summary_from_file(summary_file_path, logger=None):
    """Print console summary from saved statistics."""
    if logger is None:
        logger = logging.getLogger("get_metrics")

    stats = torch.load(summary_file_path, map_location="cpu", weights_only=False)

    metadata = stats["metadata"]
    methods = stats["methods"]

    logger.info(f"✅ Loaded statistics from: {summary_file_path}")
    logger.info(
        f"📊 Robot: {metadata['robot_name']}, Confidence: {metadata['confidence_level']}, Boundary: {metadata['boundary_points']}"
    )

    # Get data counts
    n_validation = metadata["n_validation_runs"]
    n_calibration = count_calibration_data(metadata["robot_name"])

    logger.info(
        f"📁 Data counts: {n_calibration} calibration, {n_validation} validation"
    )
    logger.info(f"📅 Generated: {metadata['timestamp'][:19]}")

    failure_rate = metadata["failure_rate"]
    target_coverage = 1.0 - failure_rate
    target_coverage_percent = target_coverage * 100

    # Load scaling factors for all methods
    scaling_factors = load_scaling_factors(
        metadata["robot_name"],
        metadata["confidence_level"],
        metadata["boundary_points"],
    )

    for method, data in methods.items():
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        scaling_factor = scaling_factors.get(method, "N/A")
        logger.info(f"\n── {display_name}  (files: {data['count']})")
        logger.info(f"{'Scaling Factor':35s}: {scaling_factor}")

        # Print regular metrics (non-timing)
        for key, mean_val in data["mean"].items():
            if key not in TIMING_METRICS:
                std_val = data["std"].get(key, 0.0)
                count_val = data["count"]

                if key == "empirical_coverage_SS":
                    coverage_proportion = mean_val  # Should be between 0 and 1
                    coverage_percent = coverage_proportion * 100
                    logger.info(
                        f"{key:35s}: {coverage_percent:10.1f}%   (N={n_validation})"
                    )
                    coverage_values = data["raw_values"].get("empirical_coverage_SS")
                    if coverage_values is None:
                        raise ValueError(
                            f"Missing empirical_coverage_SS raw values for method {method}"
                        )
                else:
                    logger.info(
                        f"{key:35s}: {mean_val:10.6f}  ± {std_val:10.6f}   (N={count_val})"
                    )

        # Print runtime metrics section
        logger.info(f"\n── Runtime Metrics ({display_name}) ──")

        # Special handling for calibration time
        if "calibration_total" in data["mean"]:
            logger.info(
                f"{'Calibration Time (s)':35s}: {data['mean']['calibration_total']:10.6f}  (single run)"
            )

        # Print other timing metrics with mean ± std across validation runs
        validation_timing_metrics = [
            key for key in TIMING_METRICS if key != "calibration_total"
        ]
        for key in validation_timing_metrics:
            if key in data["mean"]:
                mean_val = data["mean"][key]
                std_val = data["std"].get(key, 0.0)
                # Format the key name for display
                display_name = key.replace("_", " ").replace(" total", "").title()
                # Remove the trailing 's' if it exists, then add (s) for all timing metrics
                if display_name.endswith(" S"):
                    display_name = display_name[:-2]
                display_name += " (s)"
                logger.info(f"{display_name:35s}: {mean_val:10.6f}  ± {std_val:10.6f}")

    logger.info(
        f"\nN validation cases (unique base runs): {metadata['n_validation_runs']}"
    )


def generate_latex_table_from_summary(summary_file_path, logger=None):
    """Generate LaTeX table from summary statistics."""
    if logger is None:
        logger = logging.getLogger("get_metrics")
    stats = torch.load(summary_file_path, map_location="cpu", weights_only=False)

    metadata = stats["metadata"]
    methods = stats["methods"]
    volume_ratios = stats["comparisons"]["volume_ratios"]

    # Use full method display names from full_run.py
    method_display_names = METHOD_DISPLAY_NAMES

    method_order = [
        "BASELINE5",
        "BASELINE4",
        "BASELINE3",
        "BASELINE2",
        "BASELINE1",
        "BASELINE6",
        "BASELINE7",
        "CLAPS",
    ]

    # Extract metrics for each method
    table_data = []
    for method in method_order:
        if method not in methods:
            continue

        method_data = methods[method]

        # Extract averaged metrics
        marginal_coverage = (
            method_data["mean"].get("empirical_coverage_SS", 0.0) * 100
        )  # Convert to percentage
        iou_vs_mc = method_data["mean"].get("iou_vs_mc", 0.0)

        # For volume and area ratios, use individual ratios approach
        if method == "CLAPS":
            # CLAPS is the reference, so ratios are 1.0
            volume_ratio = 1.0
            area_ratio = 1.0
        else:
            # Use individual ratios from volume_ratios data
            volume_ratio_key = f"{method}_vs_CLAPS"
            if volume_ratio_key in volume_ratios:
                volume_ratio = sum(volume_ratios[volume_ratio_key]) / len(
                    volume_ratios[volume_ratio_key]
                )
            else:
                volume_ratio = 0.0

            claps_area = methods["CLAPS"]["mean"].get("area_2d", 0.0)
            method_area = method_data["mean"].get("area_2d", 0.0)
            if claps_area > 0:
                area_ratio = method_area / claps_area
            else:
                area_ratio = 0.0

        display_name = method_display_names.get(method, method)

        table_data.append(
            {
                "method": display_name,
                "coverage": marginal_coverage,
                "volume_ratio": volume_ratio,
                "area_ratio": area_ratio,
                "iou_vs_mc": iou_vs_mc,
            }
        )

    if not table_data:
        logger.warning("No valid metrics found for table generation")
        return None

    # Generate the LaTeX table using existing function logic
    failure_rate = metadata.get("failure_rate", 0.10)
    n_validation = metadata.get("n_validation_runs", None)
    robot_name = metadata["robot_name"]
    latex_table = generate_latex_table_core(
        table_data, failure_rate, n_validation, robot_name
    )

    # Save LaTeX file
    confidence_level = metadata["confidence_level"]
    boundary_points = metadata["boundary_points"]

    report_dir = Path(
        f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}"
    )
    report_dir.mkdir(exist_ok=True, parents=True)

    tex_file = report_dir / "results_table.tex"
    with open(tex_file, "w") as f:
        f.write(latex_table)
    logger.info(f"✅ LaTeX table saved to: {tex_file}")

    # Convert to PDF
    pdf_file = report_dir / "results_table.pdf"
    logger.info(f"🔄 Attempting PDF generation: {pdf_file}")
    if convert_latex_to_pdf(latex_table, pdf_file, logger):
        logger.info(f"✅ PDF table saved to: {pdf_file}")
    else:
        logger.warning(
            f"⚠️  PDF generation failed, but LaTeX file is available: {tex_file}"
        )

    return latex_table


def generate_violin_plots_from_summary(summary_file_path, logger=None):
    """Generate violin plots from summary."""
    if logger is None:
        logger = logging.getLogger("get_metrics")
    stats = torch.load(summary_file_path, map_location="cpu", weights_only=False)

    metadata = stats["metadata"]
    volume_ratios = stats["comparisons"]["volume_ratios"]
    methods_data = stats["methods"]

    if not volume_ratios:
        logger.warning("❌ No volume ratio data found. Skipping violin plots.")
        return

    robot_name = metadata["robot_name"]
    confidence_level = metadata["confidence_level"]
    boundary_points = metadata["boundary_points"]
    failure_rate = metadata.get("failure_rate", 0.10)
    target_coverage = 1.0 - failure_rate

    logger.info(
        f"📊 Loaded config: failure_rate={failure_rate}, target_coverage={target_coverage:.3f}"
    )

    # Use the same directory as summary statistics
    report_dir = Path(
        f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}"
    )
    report_dir.mkdir(exist_ok=True, parents=True)

    # Always use CLAPS as reference for all robots
    reference_suffix = "_vs_CLAPS"
    reference_label = "CLAPS"
    y_label = "Volume Ratio (Method / CLAPS)"

    # Prepare data - reverse order: BASELINE7 to BASELINE1
    baseline_order = [
        "BASELINE7",
        "BASELINE6",
        "BASELINE5",
        "BASELINE4",
        "BASELINE3",
        "BASELINE2",
        "BASELINE1",
    ]
    # Don't include CLAPS in violin plot since it would always be 1.0 (comparing to itself)
    ordered_methods = [
        m for m in baseline_order if f"{m}{reference_suffix}" in volume_ratios
    ]

    if not ordered_methods:
        logger.warning(
            f"❌ No methods vs {reference_label} comparisons found. Skipping violin plots."
        )
        return

    # Create violin plot (similar to existing logic)
    n_methods = len(ordered_methods)
    fig, ax = plt.subplots(figsize=(max(8, 2 * n_methods), 6))

    # Prepare data for violin plot
    violin_data = []
    method_names = []
    positions = []
    colors = []

    # Use centralized color scheme
    method_colors_dict = METHOD_COLORS

    # Check coverage for each method to determine which ones fail
    undercovering_methods = set()
    target_coverage_percent = (
        target_coverage * 100
    )  # Convert to percentage for comparison

    for method in ordered_methods:
        if method in methods_data:
            empirical_coverage = (
                methods_data[method]["mean"].get("empirical_coverage_SS", 0.0) * 100
            )  # Convert to percentage
            if empirical_coverage < target_coverage_percent:
                undercovering_methods.add(method)
                logger.debug(
                    f"Method {method} undercovers: {empirical_coverage:.1f}% < {target_coverage_percent:.1f}%"
                )

    for i, method in enumerate(ordered_methods):
        ratios = volume_ratios[f"{method}{reference_suffix}"]
        violin_data.append(ratios)
        method_names.append(method)
        positions.append(i + 1)
        colors.append(method_colors_dict.get(method, "tab:gray"))

    # Create violin plot
    parts = ax.violinplot(
        violin_data, positions=positions, showmeans=True, showmedians=False, widths=0.7
    )

    # Color the violin plots
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.5)

    if "cbars" in parts:
        parts["cbars"].set_color("black")
    if "cmaxes" in parts:
        parts["cmaxes"].set_color("black")
    if "cmins" in parts:
        parts["cmins"].set_color("black")

    # Use log scale for better visualization if range is large
    ax.set_yscale("log")
    
    # Custom log formatter to show standard numbers (0.1, 1, 10, 100) instead of powers of 10
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())

    # Add horizontal lines for min and max values with text labels
    for i, method in enumerate(ordered_methods):
        ratios = np.array(volume_ratios[f"{method}{reference_suffix}"])
        min_val = np.min(ratios)
        mean_val = np.mean(ratios)
        max_val = np.max(ratios)
        x_pos = i + 1
        
        # Calculate offset factor for log scale text placement
        # Log-space offset: multiply/divide by a factor instead of adding/subtracting constant
        offset_factor_text = 1.2  # 20% shift up/down
        offset_factor_line = 0.35 # Width of hlines

        # Min line
        ax.hlines(
            min_val,
            x_pos - offset_factor_line,
            x_pos + offset_factor_line,
            colors="black",
            linewidth=1.5,
            alpha=0.8,
        )

        # Max line
        ax.hlines(
            max_val,
            x_pos - offset_factor_line,
            x_pos + offset_factor_line,
            colors="black",
            linewidth=1.5,
            alpha=0.8,
        )

        # Text positioning - simpler and robust to overlap
        # Use transData for y-axis (log) but keep offsets relative
        
        ax.text(
            x_pos,
            min_val / offset_factor_text,
            f"Min: {min_val:.1f}",
            ha="center",
            va="top",
            fontsize=7,
            color="black",
        )
        
        # Mean box
        ax.text(
            x_pos,
            mean_val,
            f"Mean: {mean_val:.1f}",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.text(
            x_pos,
            max_val * offset_factor_text,
            f"Max: {max_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
        )

    # Add coverage failure indicators for undercovering methods
    for i, method in enumerate(ordered_methods):
        if method in undercovering_methods:
            x_pos = i + 1
            ratios = np.array(volume_ratios[f"{method}{reference_suffix}"])
            max_val = np.max(ratios)
            
            # Position significantly above the Max text to avoid overlap
            # In log scale, multiplying by ~2.0 gives good clearance above the max val text
            text_y_pos = max_val * 2.0

            # Add red hatched overlay over the entire violin height
            # Note: fill_between works in data coordinates, so log scale is handled naturally
            min_r = np.min(ratios)
            max_r = np.max(ratios)
            
            # Add buffer for visibility
            y_lower = min_r * 0.9
            y_upper = max_r * 1.1
            
            ax.fill_between(
                [x_pos - 0.4, x_pos + 0.4],
                y_lower,
                y_upper,
                color="red",
                alpha=0.1, # Lighter alpha
                hatch="///",
                edgecolor="red",
                linewidth=0.5,
            )

            # Add "Undercovers" text
            ax.text(
                x_pos,
                text_y_pos,
                "UNDERCOVERS",
                ha="center",
                va="bottom",
                fontsize=7,
                color="red",
                weight="bold",
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5)
            )

    # Add horizontal line at ratio = 1.0
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    
    # Adjust text position for log scale
    ax.text(
        0.02, # Left aligned
        1.05, # Just above line
        "Equal Volume (Ratio = 1)",
        transform=ax.get_yaxis_transform(), # Mix transform: x=axes, y=data
        ha="left",
        va="bottom",
        fontsize=9,
        alpha=0.7,
    )

    ax.set_xticks(positions)
    display_labels = [METHOD_DISPLAY_NAMES.get(m, m) for m in method_names]
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_ylabel(y_label, fontsize=12)

    # ax.grid(True, alpha=0.3, axis="y")
    # ax.set_axisbelow(True)

    # y_min = min([min(r) for r in violin_data])
    # y_max = max([max(r) for r in violin_data])
    # y_range = y_max - y_min
    # ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.2 * y_range)
    
    # Grid on log scale
    ax.grid(True, which="both", ls="-", alpha=0.15, axis="y")
    ax.set_axisbelow(True)
    
    # Set y-limits with padding in log space
    all_vals = np.concatenate(violin_data)
    y_min_val = np.min(all_vals)
    y_max_val = np.max(all_vals)
    
    # Add about 50% padding below and 100% padding above in log space for labels
    ax.set_ylim(y_min_val / 2.0, y_max_val * 4.0)

    plt.tight_layout()
    base_name = f"volume_ratios_violin_{robot_name}_{confidence_level}_boundary_{boundary_points}"

    violin_plot_path_pdf = report_dir / f"{base_name}.pdf"
    fig.savefig(violin_plot_path_pdf, bbox_inches="tight")
    logger.info(f"✅ Violin plot saved to: {violin_plot_path_pdf}")
    plt.close(fig)

    # Create buffered plot (exclude methods with extreme ratios > 500)
    threshold = 500.0
    filtered_methods = [
        method
        for method in ordered_methods
        if np.max(volume_ratios[f"{method}{reference_suffix}"]) <= threshold
    ]
    excluded_methods = [
        (method, np.max(volume_ratios[f"{method}{reference_suffix}"]))
        for method in ordered_methods
        if method not in filtered_methods
    ]

    if excluded_methods and filtered_methods:
        logger.info(
            f"📊 Creating buffered violin plot excluding {len(excluded_methods)} method(s) with ratio > {threshold}"
        )
        # TODO: Implement buffered plot using the same helper function approach
        logger.info(
            f"   Excluded: {[f'{m} (max={r:.1f})' for m, r in excluded_methods]}"
        )
    else:
        logger.info(
            f"ℹ️  No buffered plot needed (excluded: {len(excluded_methods)}, remaining: {len(filtered_methods)})"
        )


def generate_iou_ratio_violin_plots(summary_file_path, logger=None):
    """Generate violin plots for IoU ratios from summary."""
    if logger is None:
        logger = logging.getLogger("get_metrics")
    stats = torch.load(summary_file_path, map_location="cpu", weights_only=False)

    metadata = stats["metadata"]
    iou_ratios = stats["comparisons"].get("iou_ratios", {})
    methods_data = stats["methods"]

    if not iou_ratios:
        logger.warning("❌ No IoU ratio data found. Skipping IoU violin plots.")
        return

    robot_name = metadata["robot_name"]
    confidence_level = metadata["confidence_level"]
    boundary_points = metadata["boundary_points"]
    failure_rate = metadata.get("failure_rate", 0.10)
    target_coverage = 1.0 - failure_rate

    logger.info(
        f"📊 Loaded config for IoU plots: failure_rate={failure_rate}, target_coverage={target_coverage:.3f}"
    )

    # Use the same directory as summary statistics
    report_dir = Path(
        f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}"
    )
    report_dir.mkdir(exist_ok=True, parents=True)

    # Always use CLAPS as reference for all robots
    reference_suffix = "_vs_CLAPS"
    reference_label = "CLAPS"
    y_label = "IoU Ratio (Method / CLAPS)"

    baseline_order = [
        "BASELINE7",
        "BASELINE6",
        "BASELINE5",
        "BASELINE4",
        "BASELINE3",
        "BASELINE2",
        "BASELINE1",
    ]
    ordered_methods = [
        m for m in baseline_order if f"{m}{reference_suffix}" in iou_ratios
    ]

    if not ordered_methods:
        logger.warning(
            f"❌ No methods vs {reference_label} IoU comparisons found. Skipping IoU violin plots."
        )
        return

    # Create violin plot
    n_methods = len(ordered_methods)
    fig, ax = plt.subplots(figsize=(max(8, 2 * n_methods), 6))

    # Prepare data for violin plot
    violin_data = []
    method_names = []
    positions = []
    colors = []

    # Use centralized color scheme
    method_colors_dict = METHOD_COLORS

    # Check coverage for each method to determine which ones fail
    undercovering_methods = set()
    target_coverage_percent = (
        target_coverage * 100
    )  # Convert to percentage for comparison

    for method in ordered_methods:
        if method in methods_data:
            empirical_coverage = (
                methods_data[method]["mean"].get("empirical_coverage_SS", 0.0) * 100
            )  # Convert to percentage
            if empirical_coverage < target_coverage_percent:
                undercovering_methods.add(method)
                logger.debug(
                    f"Method {method} undercovers: {empirical_coverage:.1f}% < {target_coverage_percent:.1f}%"
                )

    for i, method in enumerate(ordered_methods):
        ratios = iou_ratios[f"{method}{reference_suffix}"]
        violin_data.append(ratios)
        method_names.append(method)
        positions.append(i + 1)
        colors.append(method_colors_dict.get(method, "tab:gray"))

    # Create violin plot (show means instead of medians)
    parts = ax.violinplot(
        violin_data, positions=positions, showmeans=True, showmedians=False, widths=0.7
    )

    # Color the violin plots
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    # Customize violin plot components (mean instead of median)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.5)

    # Set all violin vertical elements to black
    if "cbars" in parts:
        parts["cbars"].set_color("black")
    if "cmaxes" in parts:
        parts["cmaxes"].set_color("black")
    if "cmins" in parts:
        parts["cmins"].set_color("black")

    # Add horizontal lines for min and max values with text labels
    for i, method in enumerate(ordered_methods):
        ratios = np.array(iou_ratios[f"{method}{reference_suffix}"])
        min_val = np.min(ratios)
        mean_val = np.mean(ratios)  # Changed from median to mean
        max_val = np.max(ratios)
        x_pos = i + 1

        # Min line
        ax.hlines(
            min_val,
            x_pos - 0.35,
            x_pos + 0.35,
            colors="black",
            linewidth=1.5,
            alpha=0.8,
        )

        # Max line
        ax.hlines(
            max_val,
            x_pos - 0.35,
            x_pos + 0.35,
            colors="black",
            linewidth=1.5,
            alpha=0.8,
        )

        # Add text labels - centered on violin to avoid overlap
        # Min value - below violin with fixed spacing
        ax.text(
            x_pos,
            min_val - 0.15,  # Fixed spacing below
            f"Min: {min_val:.1f}",
            ha="center",
            va="top",
            fontsize=7,
            color="black",
        )

        # Mean - positioned at fixed height 2.0
        ax.text(
            x_pos,
            5.0,
            f"Mean: {mean_val:.1f}",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Max value - above violin with fixed spacing
        ax.text(
            x_pos,
            max_val + 0.15,  # Fixed spacing above
            f"Max: {max_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
        )

    # Add coverage failure indicators for undercovering methods
    for i, method in enumerate(ordered_methods):
        if method in undercovering_methods:
            x_pos = i + 1
            ratios = np.array(iou_ratios[f"{method}{reference_suffix}"])
            min_val = np.min(ratios)
            max_val = np.max(ratios)

            # Add red hatched overlay over the entire violin height
            ax.fill_between(
                [x_pos - 0.4, x_pos + 0.4],
                min_val - 0.1,  # Slightly below minimum
                max_val + 0.1,  # Slightly above maximum
                color="red",
                alpha=0.2,
                hatch="///",
                edgecolor="red",
                linewidth=0.5,
            )

            # Add "Undercovers" text above the violin
            ax.text(
                x_pos,
                max_val + 0.35,  # Position above the max value text
                "UNDERCOVERS",
                ha="center",
                va="bottom",
                fontsize=7,
                color="red",
                weight="bold",
            )

    # Add horizontal line at ratio = 1.0
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(
        0.17,
        1.05,
        "Equal IoU (Ratio = 1)",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=9,
        alpha=0.7,
    )

    ax.set_xticks(positions)
    display_labels = [METHOD_DISPLAY_NAMES.get(m, m) for m in method_names]
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_ylabel(y_label, fontsize=12)

    # Add grid
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    # Format y-axis
    y_min = min([min(r) for r in violin_data])
    y_max = max([max(r) for r in violin_data])
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.2 * y_range)

    # Save plot in PDF format
    plt.tight_layout()
    base_name = (
        f"iou_ratios_violin_{robot_name}_{confidence_level}_boundary_{boundary_points}"
    )

    violin_plot_path_pdf = report_dir / f"{base_name}.pdf"
    fig.savefig(violin_plot_path_pdf, bbox_inches="tight")
    logger.info(f"✅ IoU ratio violin plot saved to: {violin_plot_path_pdf}")
    plt.close(fig)


def generate_2d_plots_from_data(
    robot_name, confidence_level, boundary_points, summary_file_path=None, dry_run=False
):
    """Generate 2D plots from validation data."""
    import itertools

    import cv2
    from matplotlib import rcParams
    from matplotlib.ticker import MaxNLocator

    rcParams["axes.labelpad"] = 0.2

    # Load validation data for plotting
    experiment_dir = Path(
        f"data/{robot_name}/experiments/{confidence_level}/boundary_{boundary_points}"
    )
    if not experiment_dir.exists():
        print(f"No experiment directory found: {experiment_dir}")
        return

    # Load polygon data from summary statistics if available
    polygon_data = {}
    if summary_file_path and Path(summary_file_path).exists():
        try:
            summary_stats = torch.load(
                summary_file_path, map_location="cpu", weights_only=False
            )
            polygon_data = summary_stats.get("polygon_data", {})
            print(
                f"      🔍 DEBUG: Loaded polygon data for {len(polygon_data)} methods from summary"
            )
        except Exception as e:
            print(f"      ⚠️ Warning: Failed to load polygon data from summary: {e}")

    # Group metrics by base validation run
    by_base = defaultdict(dict)

    for method_dir in experiment_dir.iterdir():
        if not method_dir.is_dir():
            continue

        method_name = method_dir.name
        validation_metrics_dir = method_dir / "validation" / "metrics"

        if not validation_metrics_dir.exists():
            continue

        # Load metrics for plotting
        for metrics_file in sorted(validation_metrics_dir.glob("*.pt")):
            try:
                base_id = metrics_file.stem.replace(f"_{method_name}", "")
                data = torch.load(
                    metrics_file,
                    map_location="cpu",
                    weights_only=False,
                )

                # Merge polygon data from summary if available
                if method_name in polygon_data and base_id in polygon_data[method_name]:
                    polygon_info = polygon_data[method_name][base_id]
                    data["polygon_coords"] = polygon_info.get("coords")
                    data["area_2d"] = polygon_info.get("area", 0.0)
                    data["iou_vs_mc"] = polygon_info.get("iou_vs_mc", 0.0)
                    print(
                        f"      🔍 DEBUG: Merged polygon data for {method_name} {base_id}: area={data['area_2d']:.6f}"
                    )

                by_base[base_id][method_name] = data
            except Exception as e:
                print(f"Warning: failed to load {metrics_file}: {e}")

    if not by_base:
        print("No validation data found for 2D plots")
        return

    # Create plots directory
    save_path = Path(
        f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}/plots_2d"
    )
    save_path.mkdir(exist_ok=True, parents=True)

    # Clear existing plots to avoid overlap of old and new data
    old_plots = list(save_path.glob("*.png"))
    if old_plots:
        for old_plot in old_plots:
            old_plot.unlink()
        print(f"🧹 Cleared {len(old_plots)} old plots from: {save_path}")
    else:
        print(f"📁 No old plots to clear in: {save_path}")

    sorted_runs = sorted(by_base.items())

    # Limit runs for dry-run mode
    if dry_run:
        sorted_runs = sorted_runs[:5]  # Only process first 5 validation runs in dry-run
        print(f"🔬 DRY-RUN MODE: Processing {len(sorted_runs)} validation runs")
    else:
        print(f"🔍 DEBUG: Processing {len(sorted_runs)} validation runs")
    print(
        f"🔍 DEBUG: Available methods: {list(by_base[list(by_base.keys())[0]].keys()) if by_base else 'No data'}"
    )

    for idx, (base_id, method_map) in enumerate(sorted_runs):
        # Get shared run info
        any_d = next(iter(method_map.values()))
        run_q0 = any_d.get("q0", torch.tensor([float("nan")] * 3))
        run_dq0 = any_d.get("dq0", torch.tensor([float("nan")] * 3))
        run_u = any_d.get("u", torch.tensor([float("nan")] * 2))

        # Load MC particles from raw validation data
        raw_validation_file = Path(
            f"data/{robot_name}/raw_data/validation/{base_id}.pt"
        )
        mc_particles = None
        if raw_validation_file.exists():
            try:
                raw_data = torch.load(
                    raw_validation_file, map_location="cpu", weights_only=False
                )
                mc_particles = raw_data.get("q1")  # Shape: [100000, 3]
                print(
                    f"      🔍 DEBUG: Loaded MC particles for {base_id}: shape {mc_particles.shape if mc_particles is not None else 'None'}"
                )
            except Exception as e:
                print(
                    f"      ⚠️ Warning: Failed to load raw validation data for {base_id}: {e}"
                )
        else:
            print(
                f"      ⚠️ Warning: Raw validation file not found: {raw_validation_file}"
            )

        if mc_particles is None:
            print(f"Skipping {base_id}: no MC particles available")
            continue

        # Generate all plot versions: scatter, contour, and angle_filtered - each with buffered and non-buffered variants
        plot_configs = [
            (
                "scatter",
                False,
                False,
                False,
                False,
            ),  # (suffix, use_cv2, use_buffered_limits, use_angle_filter, show_mc_polygon)
            ("scatter_buffered", False, True, False, False),
            ("contour", True, False, False, False),
            ("contour_buffered", True, True, False, False),
            ("angle_filtered", False, False, True, False),
            ("angle_filtered_buffered", False, True, True, False),
            ("mc_polygon_buffered", False, True, False, True),
        ]

        for (
            plot_suffix,
            use_cv2,
            use_buffered_limits,
            use_angle_filter,
            show_mc_polygon,
        ) in plot_configs:
            fig, ax = plt.subplots(figsize=(7, 7), dpi=FIGURE_DPI)
            ax.set_axisbelow(True)
            ax.grid(True, linestyle="--", alpha=0.3, which="both")

            # Set axis labels to match 3D plot style
            ax.set_xlabel("X (m)", fontsize=10)
            ax.set_ylabel("Y (m)", fontsize=10)

            # Calculate reference angle for angle filtering
            reference_angle = None
            angle_threshold = 0.02
            if use_angle_filter and mc_particles is not None:
                # Check if mc_particles has proper dimensions for angle filtering
                if (
                    hasattr(mc_particles, "ndim")
                    and mc_particles.ndim >= 2
                    and mc_particles.shape[1] >= 3
                ) or (
                    hasattr(mc_particles, "dim")
                    and mc_particles.dim() >= 2
                    and mc_particles.shape[1] >= 3
                ):
                    # Use mean theta angle from Monte Carlo samples
                    reference_angle = mc_particles[:, 2].mean().item()
                    angle_range = (
                        mc_particles[:, 2].max().item()
                        - mc_particles[:, 2].min().item()
                    )
                    print(
                        f"   🔍 Angle filtering: ref={reference_angle:.3f}±{angle_threshold:.3f} rad (MC range={angle_range:.3f})"
                    )
                else:
                    print(
                        f"   ⚠️ mc_particles shape {mc_particles.shape} insufficient for angle filtering - skipping angle filter"
                    )
                    use_angle_filter = False
                    reference_angle = None

            # Collect points for buffered axis limits (exclude PP methods: BASELINE2, BASELINE3)
            buffered_methods = ["CLAPS", "BASELINE1", "BASELINE4", "BASELINE5"]
            all_buffered_points = []

            # Plot each method (skip if showing MC polygon)
            for method, d in method_map.items():
                # Skip boundary plotting when showing MC polygon
                if show_mc_polygon:
                    continue

                boundary = d.get("boundary_SS_pts")
                if boundary is None:
                    print(
                        f"      🔍 DEBUG: {method} - NO boundary_SS_pts data, skipping"
                    )
                    continue
                print(
                    f"      🔍 DEBUG: {method} - boundary shape: {boundary.shape if hasattr(boundary, 'shape') else type(boundary)}"
                )

                # Ensure boundary is 2D (sometimes can be 1D due to data issues)
                if hasattr(boundary, "ndim"):
                    if boundary.ndim == 1:
                        print(
                            f"      {method}: WARNING - boundary is 1D with shape {boundary.shape}, skipping"
                        )
                        continue
                elif hasattr(boundary, "dim"):
                    if boundary.dim() == 1:
                        print(
                            f"      {method}: WARNING - boundary is 1D with shape {boundary.shape}, skipping"
                        )
                        continue

                color = METHOD_COLORS.get(method, "tab:gray")

                # Apply angle filtering if requested
                if use_angle_filter and reference_angle is not None:
                    boundary_filtered, mask = filter_data_by_angle(
                        boundary, reference_angle, angle_threshold
                    )
                    if boundary_filtered is not None and len(boundary_filtered) > 0:
                        # Additional safety check for 2D shape
                        if boundary_filtered.ndim == 1 or (
                            hasattr(boundary_filtered, "dim")
                            and boundary_filtered.dim() == 1
                        ):
                            print(
                                f"      {method}: WARNING - boundary_filtered is 1D, reshaping to 2D"
                            )
                            boundary_filtered = (
                                boundary_filtered.reshape(1, -1)
                                if hasattr(boundary_filtered, "reshape")
                                else boundary_filtered.unsqueeze(0)
                            )
                        points_2d = boundary_filtered[:, :2]
                        filtered_angles = boundary_filtered[:, 2]
                        print(
                            f"      {method}: {len(boundary_filtered)}/{len(boundary)} pts after filtering "
                            f"(θ range: {filtered_angles.min():.3f} to {filtered_angles.max():.3f})"
                        )
                    else:
                        # Skip this method if no points after filtering
                        print(
                            f"      {method}: 0/{len(boundary)} pts after filtering - SKIPPING"
                        )
                        continue
                else:
                    # Check if boundary has proper dimensions
                    if (
                        hasattr(boundary, "ndim")
                        and boundary.ndim >= 2
                        and boundary.shape[1] >= 2
                    ) or (
                        hasattr(boundary, "dim")
                        and boundary.dim() >= 2
                        and boundary.shape[1] >= 2
                    ):
                        points_2d = boundary[:, :2]
                    else:
                        print(
                            f"      {method}: WARNING - boundary shape {boundary.shape} insufficient, skipping"
                        )
                        continue

                # Collect points for buffered axis limits
                if use_buffered_limits and method in buffered_methods:
                    all_buffered_points.append(points_2d)

                # Check if we have pre-computed polygon coordinates
                polygon_coords = d.get("polygon_coords")
                if use_cv2 and polygon_coords is not None:
                    # Use actual fitted polygon coordinates
                    try:
                        # Convert to numpy if needed
                        if hasattr(polygon_coords, "cpu"):
                            coords = polygon_coords.cpu().numpy()
                        elif hasattr(polygon_coords, "numpy"):
                            coords = polygon_coords.numpy()
                        else:
                            coords = np.array(polygon_coords)

                        # Ensure coordinates are closed (first point = last point)
                        if not np.allclose(coords[0], coords[-1]):
                            coords = np.vstack([coords, coords[0]])

                        linestyle = get_method_linestyle(method)

                        display_name = METHOD_DISPLAY_NAMES.get(method, method)
                        ax.plot(
                            coords[:, 0],
                            coords[:, 1],
                            color=color,
                            alpha=0.9,
                            linewidth=2.5,
                            linestyle=linestyle,
                            label=display_name,
                        )
                        ax.fill(
                            coords[:, 0],
                            coords[:, 1],
                            color=color,
                            alpha=0.1,
                        )
                    except Exception as e:
                        print(
                            f"      {method}: Failed to use polygon_coords ({e}), falling back to scatter"
                        )
                        # Fallback to scatter
                        display_name = METHOD_DISPLAY_NAMES.get(method, method)
                        ax.scatter(
                            points_2d[:, 0],
                            points_2d[:, 1],
                            s=2,
                            alpha=0.5,
                            color=color,
                            label=display_name,
                        )
                elif use_cv2 and len(points_2d) >= 10:
                    print(
                        f"      {method}: No polygon_coords available, falling back to scatter"
                    )
                    # Fallback to scatter when no polygon coordinates available
                    display_name = METHOD_DISPLAY_NAMES.get(method, method)
                    ax.scatter(
                        points_2d[:, 0],
                        points_2d[:, 1],
                        s=2,
                        alpha=0.5,
                        color=color,
                        label=display_name,
                    )
                else:
                    # Scatter plot (when use_cv2 is False)
                    display_name = METHOD_DISPLAY_NAMES.get(method, method)
                    ax.scatter(
                        points_2d[:, 0],
                        points_2d[:, 1],
                        s=4,
                        alpha=0.7,
                        color=color,
                        label=display_name,
                    )

            # Plot Monte Carlo particles
            if mc_particles is not None and mc_particles.shape[0] > 0:
                # Apply angle filtering to MC particles if requested
                if use_angle_filter and reference_angle is not None:
                    # Filter MC particles by angle (similar to test_comparison.py)
                    mc_filtered, _ = filter_data_by_angle(
                        mc_particles, reference_angle, angle_threshold
                    )
                    if mc_filtered is not None and len(mc_filtered) > 0:
                        mc_filtered_np = (
                            mc_filtered.cpu().numpy()
                            if hasattr(mc_filtered, "cpu")
                            else mc_filtered
                        )

                        ax.scatter(
                            mc_filtered_np[:, 0],
                            mc_filtered_np[:, 1],
                            s=0.1,  # Very small points
                            alpha=0.2,
                            label="Monte Carlo Samples",
                            color="k",
                            zorder=MC_PARTICLES_ZORDER,
                        )

                        # Plot filtered MC mean
                        mc_filtered_mean_x = mc_filtered_np[:, 0].mean()
                        mc_filtered_mean_y = mc_filtered_np[:, 1].mean()
                        ax.scatter(
                            mc_filtered_mean_x,
                            mc_filtered_mean_y,
                            s=25,
                            alpha=0.9,
                            color="red",
                            zorder=160,
                            marker="x",
                            label="Monte Carlo Mean",
                        )
                        print(
                            f"      🔍 DEBUG: Plotted {len(mc_filtered)} filtered MC particles (out of {mc_particles.shape[0]} total)"
                        )
                    else:
                        print(f"      ⚠️ Warning: No MC particles after angle filtering")
                else:
                    # Plot all MC particles without angle filtering
                    mc_np = mc_particles.cpu().numpy()
                    ax.scatter(
                        mc_np[:, 0],
                        mc_np[:, 1],
                        s=0.1,  # Very small points for 100k particles
                        alpha=0.2,
                        label="Monte Carlo Samples",
                        color="k",
                        zorder=MC_PARTICLES_ZORDER,
                    )

                    # Plot MC mean
                    mc_mean_x = mc_np[:, 0].mean()
                    mc_mean_y = mc_np[:, 1].mean()
                    ax.scatter(
                        mc_mean_x,
                        mc_mean_y,
                        s=25,
                        alpha=0.9,
                        color="red",
                        zorder=160,
                        marker="x",
                        label="Monte Carlo Mean",
                    )
                    print(
                        f"      🔍 DEBUG: Plotted all {mc_particles.shape[0]} MC particles"
                    )

                # Plot MC polygon if requested (use pre-computed polygon if available)
                if show_mc_polygon:
                    # Try to get pre-computed polygon from any method's data
                    mc_polygon_data = None
                    for method, d in method_map.items():
                        if base_id in d and hasattr(d[base_id], "get"):
                            pass

                    # Use pre-computed MC polygon from summary file (must exist from metrics phase)
                    mc_coords = None
                    mc_area = None
                    if (
                        "MC_particles" in polygon_data
                        and base_id in polygon_data["MC_particles"]
                    ):
                        mc_coords = polygon_data["MC_particles"][base_id].get("coords")
                        mc_area = polygon_data["MC_particles"][base_id].get("area", 0)

                    if mc_coords is not None:
                        ax.plot(
                            mc_coords[:, 0],
                            mc_coords[:, 1],
                            color="black",
                            linewidth=1,
                            linestyle="--",
                            alpha=0.6,
                            label=f"MC Polygon (Area: {mc_area:.4f})",
                            zorder=150,
                        )
                        ax.fill(
                            mc_coords[:, 0],
                            mc_coords[:, 1],
                            color="gray",
                            alpha=0.15,
                            zorder=50,
                        )
                        print(f"      🔍 DEBUG: MC polygon area: {mc_area:.6f}")
                    else:
                        print(f"      ⚠️ Warning: Failed to create MC polygon")
            else:
                print(f"      ⚠️ Warning: No MC particles to plot")

            # Plot method polygons alongside MC polygon
            if show_mc_polygon:
                for method, d in method_map.items():
                    # Check if method has pre-computed polygon coordinates
                    polygon_coords = d.get("polygon_coords")
                    area_2d = d.get("area_2d", 0.0)

                    if polygon_coords is None or len(polygon_coords) == 0:
                        # Polygon data should have been computed during statistics phase
                        print(f"      ❌ ERROR: Polygon data not found for {method}")
                        print(
                            f"      💡 Please regenerate statistics with: python scripts/get_metrics.py --robot_type {robot_name} --confidence-level {confidence_level} --boundary-points {boundary_points}"
                        )
                        continue

                    # Collect polygon coordinates for buffered axis limits (exclude PP methods)
                    if use_buffered_limits and method in buffered_methods:
                        all_buffered_points.append(polygon_coords)

                    # Get method color
                    color = METHOD_COLORS.get(method, "tab:gray")
                    display_name = METHOD_DISPLAY_NAMES.get(method, method)

                    # Plot polygon boundary
                    ax.plot(
                        polygon_coords[:, 0],
                        polygon_coords[:, 1],
                        color=color,
                        linewidth=1.5,
                        linestyle="-",
                        alpha=0.8,
                        label=f"{display_name} (Area: {area_2d:.4f})",
                        zorder=100,
                    )
                    # Add semi-transparent fill
                    ax.fill(
                        polygon_coords[:, 0],
                        polygon_coords[:, 1],
                        color=color,
                        alpha=0.15,
                        zorder=40,
                    )
                    print(
                        f"      🔍 DEBUG: Plotted {method} polygon area: {area_2d:.6f}"
                    )

                # Also add MC polygon coordinates to buffered points for proper zooming
                if (
                    use_buffered_limits
                    and "MC_particles" in polygon_data
                    and base_id in polygon_data["MC_particles"]
                ):
                    mc_coords = polygon_data["MC_particles"][base_id].get("coords")
                    if mc_coords is not None:
                        all_buffered_points.append(mc_coords)
                        print(
                            f"      🔍 DEBUG: Added MC polygon coords to buffered points for zooming"
                        )

            # Apply buffered axis limits if requested
            if use_buffered_limits and all_buffered_points:
                # Combine all points from buffered methods
                combined_points = np.vstack(all_buffered_points)
                x_min_buf, x_max_buf = (
                    combined_points[:, 0].min(),
                    combined_points[:, 0].max(),
                )
                y_min_buf, y_max_buf = (
                    combined_points[:, 1].min(),
                    combined_points[:, 1].max(),
                )

                # Add small padding
                x_range = x_max_buf - x_min_buf
                y_range = y_max_buf - y_min_buf
                padding = 0.05 * max(x_range, y_range)

                ax.set_xlim(x_min_buf - padding, x_max_buf + padding)
                ax.set_ylim(y_min_buf - padding, y_max_buf + padding)

            # Ensure equal data scaling on both axes (proper 2D distance plots)
            # This preserves metric distances while allowing rectangular figure shapes
            ax.set_aspect("equal", adjustable="box")

            # Formatting - handle combinations of features
            plot_type_parts = []
            if use_cv2:
                plot_type_parts.append("Contour")
            else:
                plot_type_parts.append("Scatter")

            if use_angle_filter and reference_angle is not None:
                plot_type_parts.append(
                    f"Angle Filtered (θ≈{reference_angle:.1f}±{angle_threshold:.1f})"
                )

            if use_buffered_limits:
                plot_type_parts.append("Buffered (CLAPS, ABL1,4,5 limits)")

            if show_mc_polygon:
                plot_type_parts.append("MC Polygon")

            plot_type_label = ", ".join(plot_type_parts)

            # Save figure with dq0 and u info in filename
            dq0_str = format_tensor_for_filename(run_dq0)
            u_str = format_tensor_for_filename(run_u)
            filename = (
                f"fig_2D_projection_{base_id}_{plot_suffix}_dq0_{dq0_str}_u_{u_str}"
            )
            plt.tight_layout()
            figure_base_path = save_path / filename
            plt.savefig(figure_base_path.with_suffix('.png'), dpi=FIGURE_DPI)
            plt.close(fig)

    print(f"✅ 2D plots saved to: {save_path}")


def generate_3d_plots_from_data_paper_figure(
    robot_name, confidence_level, boundary_points
):
    """Generate 3D plots from validation data."""
    from mpl_toolkits.mplot3d import Axes3D

    # Load validation data
    experiment_dir = Path(
        f"data/{robot_name}/experiments/{confidence_level}/boundary_{boundary_points}"
    )
    if not experiment_dir.exists():
        print(f"No experiment directory found: {experiment_dir}")
        return

    # Get list of validation IDs from any method's metrics folder
    validation_ids = set()
    raw_validation_dir = Path(f"data/{robot_name}/raw_data/validation")

    # Find available validation IDs by scanning one method's metrics folder
    first_method_dir = None
    for method_dir in experiment_dir.iterdir():
        if method_dir.is_dir() and (method_dir / "validation" / "metrics").exists():
            first_method_dir = method_dir
            break

    if not first_method_dir:
        print("No methods with validation metrics found")
        return

    metrics_dir = first_method_dir / "validation" / "metrics"
    method_name = first_method_dir.name

    # Extract base IDs from metrics files
    for i, metrics_file in enumerate(sorted(metrics_dir.glob("*.pt"))):
        base_id = metrics_file.stem.replace(f"_{method_name}", "")
        # Only process validation ID 301 as requested
        if "301" in base_id:
            validation_ids.add(base_id)

    if not validation_ids:
        print("No validation IDs found (looking for *301*)")
        return

    print(f"Found {len(validation_ids)} validation IDs for 3D plotting")

    # Create plots directory
    plots_3d_dir = Path(
        f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}/plots_3d"
    )
    plots_3d_dir.mkdir(exist_ok=True, parents=True)

    # Define viewing angles - only 315 as requested
    azimuth_angles = [315]

    # Process each validation ID one at a time (memory efficient)
    for base_id in sorted(validation_ids):
        # Load raw MC particles for this validation ID
        raw_file_path = raw_validation_dir / f"{base_id}.pt"
        if not raw_file_path.exists():
            print(f"Raw data file not found: {raw_file_path}")
            continue

        try:
            print(f"\n  Loading MC particles for {base_id}...")
            raw_data = torch.load(raw_file_path, map_location="cpu", weights_only=False)
            full_mc_particles = raw_data.get("q1")  # [100000, 3]

            if full_mc_particles is None:
                print(f"No q1 data found in {raw_file_path}")
                continue

            # Calculate MC particles mean
            mc_particles_mean = full_mc_particles.mean(dim=0)  # [3]
            print(
                f"    Loaded {len(full_mc_particles)} MC particles, mean: {mc_particles_mean.numpy()}"
            )

        except Exception as e:
            print(f"Failed to load raw data from {raw_file_path}: {e}")
            continue

        # Load metrics for all methods for this validation ID
        method_data = {}
        for method_dir in experiment_dir.iterdir():
            if not method_dir.is_dir():
                continue

            method_name = method_dir.name
            validation_metrics_dir = method_dir / "validation" / "metrics"
            if not validation_metrics_dir.exists():
                continue

            # Look for this specific validation ID's metrics file
            metrics_file = validation_metrics_dir / f"{base_id}_{method_name}.pt"
            if metrics_file.exists():
                try:
                    data = torch.load(
                        metrics_file, map_location="cpu", weights_only=False
                    )
                    method_data[method_name] = data
                except Exception as e:
                    print(f"Failed to load {metrics_file}: {e}")

        if not method_data:
            print(f"No method data found for {base_id}")
            # Free memory
            del raw_data, full_mc_particles
            continue

        print(f"    Loaded metrics for methods: {list(method_data.keys())}")

        # Check coverage criteria: BASELINE1 < 90% AND CLAPS > 90%
        if "BASELINE1" in method_data and "CLAPS" in method_data:
            baseline1_mask = method_data["BASELINE1"].get("mask_state_space")
            claps_mask = method_data["CLAPS"].get("mask_state_space")

            if baseline1_mask is not None and claps_mask is not None:
                baseline1_coverage = (
                    baseline1_mask.mean().item() * 100
                )  # Convert to percentage
                claps_coverage = claps_mask.mean().item() * 100  # Convert to percentage

                print(
                    f"    Coverage: SS EKF + CP = {baseline1_coverage:.1f}%, CLAPS = {claps_coverage:.1f}%"
                )

                # Skip if criteria not met
                if baseline1_coverage >= 90.0 or claps_coverage <= 90.0:
                    print(
                        f"    SKIPPING {base_id}: Coverage criteria not met (need SS EKF + CP < 90% AND CLAPS > 90%)"
                    )
                    # Free memory
                    del raw_data, full_mc_particles
                    continue
                else:
                    print(f"    PLOTTING {base_id}: Meets coverage criteria")
            else:
                print(f"    WARNING: Missing coverage masks for {base_id}")
                # Free memory
                del raw_data, full_mc_particles
                continue
        else:
            print(
                f"    SKIPPING {base_id}: Missing required methods (BASELINE1 or CLAPS)"
            )
            # Free memory
            del raw_data, full_mc_particles
            continue

        # Generate plots for each azimuth angle
        for az_val in azimuth_angles:
            # Define the desired 2x2 layout order: SS EKF, InEKF (top row), SS EKF + CP, CLAPS (bottom row)
            desired_order = [
                "BASELINE5",
                "BASELINE4",
                "BASELINE1",
                "CLAPS",
            ]

            # Filter to only methods we have data for, maintaining the desired order
            available_methods = [
                (method, method_data[method])
                for method in desired_order
                if method in method_data
            ]

            # Calculate global axis limits across all methods for consistent scaling
            global_all_points = []
            for method, d in available_methods:
                boundary_pts = d.get("boundary_SS_pts")
                if boundary_pts is not None and len(boundary_pts) > 0:
                    global_all_points.append(boundary_pts)

            # Add MC particles for global limits
            global_all_points.append(full_mc_particles)

            # Include mean points for global axis limits
            mean_points_global = np.array([mc_particles_mean.numpy()])
            for method, d in available_methods:
                model_mean_pred = d.get("mean_pred")
                if model_mean_pred is not None:
                    mean_points_global = np.vstack(
                        [mean_points_global, model_mean_pred.numpy()]
                    )
            global_all_points.append(mean_points_global)

            # Calculate global limits
            if global_all_points:
                xyz_all_global = np.vstack(global_all_points)
                global_mins, global_maxs = xyz_all_global.min(0), xyz_all_global.max(0)
                global_span = global_maxs - global_mins
                global_mins -= 0.1 * global_span  # More margin for visibility
                global_maxs += 0.1 * global_span
                print(
                    f"    Global axis limits: X[{global_mins[0]:.3f}, {global_maxs[0]:.3f}], Y[{global_mins[1]:.3f}, {global_maxs[1]:.3f}], Z[{global_mins[2]:.3f}, {global_maxs[2]:.3f}]"
                )

            # Use 2 rows, 2 columns layout; make figure slightly wider to avoid horizontal squeeze
            fig = plt.figure(figsize=(14, 12), dpi=FIGURE_DPI)

            for i, (method_name, d) in enumerate(available_methods):
                # Calculate subplot position: 2 rows, 2 columns
                subplot_pos = i + 1
                ax = fig.add_subplot(2, 2, subplot_pos, projection="3d")

                # Get method data (boundaries, masks from metrics)
                boundary_pts = d.get("boundary_SS_pts")
                coverage_mask = d.get("mask_state_space")
                model_mean_pred = d.get("mean_pred")  # Model's prediction [3]

                if boundary_pts is None or coverage_mask is None:
                    print(
                        f"      WARNING: Missing boundary or mask data for {method_name}"
                    )
                    continue

                # Use full MC particles (not the single point from metrics)
                if coverage_mask.shape[0] != full_mc_particles.shape[0]:
                    print(
                        f"      WARNING: Coverage mask size ({coverage_mask.shape[0]}) != MC particles ({full_mc_particles.shape[0]})"
                    )
                    continue

                # 1. Plot MC particles
                inside_particles = full_mc_particles[coverage_mask]
                outside_particles = full_mc_particles[~coverage_mask]

                if len(inside_particles) > 0:
                    ax.scatter(
                        inside_particles[:, 0],
                        inside_particles[:, 1],
                        inside_particles[:, 2],
                        c="green",
                        s=1,
                        alpha=0.6,
                        label="Points in",
                    )
                if len(outside_particles) > 0:
                    ax.scatter(
                        outside_particles[:, 0],
                        outside_particles[:, 1],
                        outside_particles[:, 2],
                        c="red",
                        s=1,
                        alpha=0.6,
                        label="Points out",
                    )

                # 4. Plot method boundary points
                if boundary_pts is not None and len(boundary_pts) > 0:
                    ax.scatter(
                        boundary_pts[:, 0],
                        boundary_pts[:, 1],
                        boundary_pts[:, 2],
                        c="orange",
                        s=1,
                        alpha=0.8,
                        label="Boundary pts",
                    )

                # Use global axis limits for consistent scaling across all subplots
                if global_all_points:
                    ax.set_xlim(global_mins[0], global_maxs[0])
                    ax.set_ylim(global_mins[1], global_maxs[1])
                    ax.set_zlim(global_mins[2], global_maxs[2])

                    # Use cubic box aspect to prevent horizontal squashing from tall Z ranges
                    try:
                        ax.set_box_aspect((1.0, 1.0, 1.0))
                    except Exception:
                        pass

                # Labels and formatting with rotation (minimize padding)
                ax.set_xlabel("X (m)", fontsize=12)
                ax.set_ylabel("Y (m)", fontsize=12)
                ax.set_zlabel("θ (rad)", fontsize=12)

                # Apply rotation to 3D axis labels and reduce padding
                ax.xaxis.label.set_rotation(45)
                ax.yaxis.label.set_rotation(45)
                ax.xaxis.labelpad = 0
                ax.yaxis.labelpad = 0
                ax.zaxis.labelpad = 0
                ax.tick_params(pad=1, labelsize=10)

                # Eliminate extra margins inside the axes box
                try:
                    ax.margins(x=0.0, y=0.0, z=0.0)
                except Exception:
                    pass

                # Get display name and clean it up for first row methods
                display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
                if method_name in [
                    "BASELINE3",
                    "BASELINE2",
                ]:  # First row methods: SS PP, Lie PP
                    display_name = display_name.replace(" + CP", "")

                # Use inline text instead of axis title to eliminate extra top padding
                # Clear any existing title to avoid reserved space
                ax.set_title("")

                # Bold only the method name for CLAPS using mathtext (no bold for coverage)
                if method_name == "CLAPS":
                    name_str = r"$\mathbf{" + display_name + r"}$"
                else:
                    name_str = display_name

                title_text = f"{name_str} (Coverage: {coverage_mask.mean():.1%})"
                ax.text2D(
                    0.5,
                    0.962,
                    title_text,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=12,
                )

                # Anchor axes to the top to minimize space above content
                try:
                    ax.set_anchor("N")
                except Exception:
                    pass

                ax.view_init(elev=20, azim=az_val)
                # ax.legend(loc="upper right", fontsize=7, markerscale=0.7)  # Legend removed

            # Manage whitespace; reserve room on the right for a vertical shared legend
            plt.subplots_adjust(
                left=0.0, right=1, top=1, bottom=0.0, wspace=0.01, hspace=0.01
            )
            # plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.05, wspace=0.3)  # Fixed spacing and Z-label clipping

            # Figure-level shared legend on the right
            try:
                from matplotlib.lines import Line2D

                shared_handles = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=r"$\mathcal{C}^q$ Boundary Points",
                        markerfacecolor="orange",
                        markersize=9,
                        alpha=1,
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=r"MC Samples Inside $\mathcal{C}^q$",
                        markerfacecolor="green",
                        markersize=9,
                        alpha=1,
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=r"MC Samples Outside $\mathcal{C}^q$",
                        markerfacecolor="red",
                        markersize=9,
                        alpha=1,
                    ),
                ]

                # Place legend inside figure canvas on the right, single column (vertical)
                fig.legend(
                    handles=shared_handles,
                    loc="center left",
                    bbox_to_anchor=(0.2, 0.5),
                    frameon=True,
                    fontsize=13,
                    borderaxespad=0.0,
                    ncol=3,
                )
            except Exception:
                pass

            filename = f"fig_3D_{base_id}_az{az_val:03d}.png"
            fig.savefig(plots_3d_dir / filename, dpi=FIGURE_DPI)
            fig.savefig(plots_3d_dir / filename.replace(".png", ".pdf"), dpi=FIGURE_DPI)
            fig.savefig(plots_3d_dir / filename.replace(".png", ".svg"), dpi=FIGURE_DPI)
            print(f"Saved 3D plot: {filename}")
            plt.close(fig)

        # Free memory after processing this validation ID
        del raw_data, full_mc_particles, mc_particles_mean, method_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  Memory freed for {base_id}")

    print(f"✅ 3D plots saved to: {plots_3d_dir}")


def generate_3d_plots_from_data(robot_name, confidence_level, boundary_points):
    """Generate 3D plots from validation data using Plotly for proper depth sorting."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Load validation data
    experiment_dir = Path(
        f"data/{robot_name}/experiments/{confidence_level}/boundary_{boundary_points}"
    )
    if not experiment_dir.exists():
        print(f"No experiment directory found: {experiment_dir}")
        return

    # Get list of validation IDs from any method's metrics folder
    validation_ids = set()
    raw_validation_dir = Path(f"data/{robot_name}/raw_data/validation")

    # Find available validation IDs by scanning one method's metrics folder
    first_method_dir = None
    for method_dir in experiment_dir.iterdir():
        if method_dir.is_dir() and (method_dir / "validation" / "metrics").exists():
            first_method_dir = method_dir
            break

    if not first_method_dir:
        print("No methods with validation metrics found")
        return

    metrics_dir = first_method_dir / "validation" / "metrics"
    method_name = first_method_dir.name

    # Extract base IDs from metrics files
    for i, metrics_file in enumerate(sorted(metrics_dir.glob("*.pt"))):
        base_id = metrics_file.stem.replace(f"_{method_name}", "")
        validation_ids.add(base_id)

    if not validation_ids:
        print("No validation IDs found")
        return

    print(f"Found {len(validation_ids)} validation IDs for 3D plotting")

    # Create plots directory
    plots_3d_dir = Path(
        f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}/plots_3d"
    )
    plots_3d_dir.mkdir(exist_ok=True, parents=True)

    # Define viewing angles
    azimuth_angles = list(range(0, 360, 1))

    # Process each validation ID one at a time (memory efficient)
    for base_id in sorted(validation_ids):
        # Filter by specific validation ID if requested
        if args.validation_id and base_id != f"val_isaac_{args.validation_id}":
            continue
        # Load raw MC particles for this validation ID
        raw_file_path = raw_validation_dir / f"{base_id}.pt"
        if not raw_file_path.exists():
            print(f"Raw data file not found: {raw_file_path}")
            continue

        try:
            print(f"\n  Loading MC particles for {base_id}...")
            raw_data = torch.load(raw_file_path, map_location="cpu", weights_only=False)
            full_mc_particles = raw_data.get("q1")  # [100000, 3]

            if full_mc_particles is None:
                print(f"No q1 data found in {raw_file_path}")
                continue

            # Calculate MC particles mean
            mc_particles_mean = full_mc_particles.mean(dim=0)  # [3]
            print(
                f"    Loaded {len(full_mc_particles)} MC particles, mean: {mc_particles_mean.numpy()}"
            )

        except Exception as e:
            print(f"Failed to load raw data from {raw_file_path}: {e}")
            continue

        # Load metrics for all methods for this validation ID
        method_data = {}
        for method_dir in experiment_dir.iterdir():
            if not method_dir.is_dir():
                continue

            method_name = method_dir.name
            validation_metrics_dir = method_dir / "validation" / "metrics"

            if not validation_metrics_dir.exists():
                continue

            # Look for this specific validation ID's metrics file
            metrics_file = validation_metrics_dir / f"{base_id}_{method_name}.pt"
            if metrics_file.exists():
                try:
                    data = torch.load(
                        metrics_file, map_location="cpu", weights_only=False
                    )
                    method_data[method_name] = data
                except Exception as e:
                    print(f"Failed to load {metrics_file}: {e}")

        if not method_data:
            print(f"No method data found for {base_id}")
            # Free memory
            del raw_data, full_mc_particles
            continue

        print(f"    Loaded metrics for methods: {list(method_data.keys())}")

        # Check coverage criteria: BASELINE1 < 90% AND CLAPS > 90%
        if "BASELINE1" in method_data and "CLAPS" in method_data:
            baseline1_mask = method_data["BASELINE1"].get("mask_state_space")
            claps_mask = method_data["CLAPS"].get("mask_state_space")

            if baseline1_mask is not None and claps_mask is not None:
                baseline1_coverage = (
                    baseline1_mask.mean().item() * 100
                )  # Convert to percentage
                claps_coverage = claps_mask.mean().item() * 100  # Convert to percentage

                print(
                    f"    Coverage: SS EKF + CP = {baseline1_coverage:.1f}%, CLAPS = {claps_coverage:.1f}%"
                )

                # Skip if criteria not met (unless skip-coverage-check is enabled)
                if not args.skip_coverage_check and (baseline1_coverage >= 90.0 or claps_coverage <= 90.0):
                    print(
                        f"    SKIPPING {base_id}: Coverage criteria not met (need SS EKF + CP < 90% AND CLAPS > 90%)"
                    )
                    # Free memory
                    del raw_data, full_mc_particles
                    continue
                else:
                    print(f"    PLOTTING {base_id}: Meets coverage criteria")
            else:
                print(f"    WARNING: Missing coverage masks for {base_id}")
                # Free memory
                del raw_data, full_mc_particles
                continue
        else:
            print(
                f"    SKIPPING {base_id}: Missing required methods (BASELINE1 or CLAPS)"
            )
            # Free memory
            del raw_data, full_mc_particles
            continue

        # Generate a single interactive plot per validation ID (no need for multiple angles)
        # Define the desired 2x2 layout order: SS EKF, InEKF (top row), SS EKF + CP, CLAPS (bottom row)
        desired_order = [
            "BASELINE5",
            "BASELINE4",
            "BASELINE1",
            "CLAPS",
        ]

        # Filter to only methods we have data for, maintaining the desired order
        available_methods = [
            (method, method_data[method])
            for method in desired_order
            if method in method_data
        ]

        # Calculate global axis limits across all methods for consistent scaling
        global_all_points = []
        for method, d in available_methods:
            boundary_pts = d.get("boundary_SS_pts")
            if boundary_pts is not None and len(boundary_pts) > 0:
                global_all_points.append(boundary_pts)

        # Add MC particles for global limits
        global_all_points.append(full_mc_particles)

        # Include mean points for global axis limits
        mean_points_global = np.array([mc_particles_mean.numpy()])
        for method, d in available_methods:
            model_mean_pred = d.get("mean_pred")
            if model_mean_pred is not None:
                mean_points_global = np.vstack(
                    [mean_points_global, model_mean_pred.numpy()]
                )
        global_all_points.append(mean_points_global)

        # Calculate global limits
        if global_all_points:
            xyz_all_global = np.vstack(global_all_points)
            global_mins, global_maxs = xyz_all_global.min(0), xyz_all_global.max(0)
            global_span = global_maxs - global_mins
            global_mins -= 0.1 * global_span  # More margin for visibility
            global_maxs += 0.1 * global_span
            print(
                f"    Global axis limits: X[{global_mins[0]:.3f}, {global_maxs[0]:.3f}], Y[{global_mins[1]:.3f}, {global_maxs[1]:.3f}], Z[{global_mins[2]:.3f}, {global_maxs[2]:.3f}]"
            )

        # Create Plotly subplots with 3D support
        subplot_titles = []
        for method_name, d in available_methods:
            coverage_mask = d.get("mask_state_space")
            display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
            if method_name in ["BASELINE3", "BASELINE2"]:
                display_name = display_name.replace(" + CP", "")
            if method_name == "CLAPS":
                title = f"<b>{display_name}</b> (Coverage: {coverage_mask.mean():.1%})"
            else:
                title = f"{display_name} (Coverage: {coverage_mask.mean():.1%})"
            subplot_titles.append(title)

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scatter3d"}, {"type": "scatter3d"}],
                [{"type": "scatter3d"}, {"type": "scatter3d"}],
            ],
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
        )

        # Add title and apply dark theme
        fig.update_layout(
            # template="plotly_dark",
            title=dict(
                text=f"JetBot (Simulation), Validation ID: {base_id.split('_')[-1]}<br>C-Space Visualization",
                x=0.5,
                font=dict(size=16),
            ),
            showlegend=True,
            legend=dict(
                x=0.5,  # Center horizontally
                y=0.5,  # Center vertically
                xanchor="center",  # Anchor legend center to x position
                yanchor="middle",  # Anchor legend middle to y position
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor="White",
                borderwidth=1,
                itemsizing="constant",  # Keep marker sizes constant
                font=dict(size=16),  # Increase legend font size
                orientation="h",  # Horizontal layout (3 columns instead of 3 rows)
                # Increase marker size in legend
                itemwidth=30,
                tracegroupgap=10,
            ),
            width=1200,
            height=1000,
        )

        # Process each method subplot
        for i, (method_name, d) in enumerate(available_methods):
            row = (i // 2) + 1
            col = (i % 2) + 1

            # Get method data
            boundary_pts = d.get("boundary_SS_pts")
            coverage_mask = d.get("mask_state_space")

            if boundary_pts is None or coverage_mask is None:
                print(f"      WARNING: Missing boundary or mask data for {method_name}")
                continue

            if coverage_mask.shape[0] != full_mc_particles.shape[0]:
                print(
                    f"      WARNING: Coverage mask size ({coverage_mask.shape[0]}) != MC particles ({full_mc_particles.shape[0]})"
                )
                continue

            # Split MC particles
            inside_particles = full_mc_particles[coverage_mask]
            outside_particles = full_mc_particles[~coverage_mask]

            # Add outside particles (red)
            if len(outside_particles) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=outside_particles[:, 0],
                        y=outside_particles[:, 1],
                        z=outside_particles[:, 2],
                        mode="markers",
                        marker=dict(size=1, color="red", opacity=0.4),
                        name=r"$\text{MC Samples Outside } \mathcal{C}^q$",
                        legendgroup="outside",
                        showlegend=(i == 0),  # Only show in legend once
                    ),
                    row=row,
                    col=col,
                )

            # Add inside particles (green)
            if len(inside_particles) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=inside_particles[:, 0],
                        y=inside_particles[:, 1],
                        z=inside_particles[:, 2],
                        mode="markers",
                        marker=dict(size=1, color="green", opacity=0.7),
                        name=r"$\text{MC Samples Inside } \mathcal{C}^q$",
                        legendgroup="inside",
                        showlegend=(i == 0),  # Only show in legend once
                    ),
                    row=row,
                    col=col,
                )

            # Add boundary points (orange)
            if boundary_pts is not None and len(boundary_pts) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=boundary_pts[:, 0],
                        y=boundary_pts[:, 1],
                        z=boundary_pts[:, 2],
                        mode="markers",
                        marker=dict(size=1.5, color="orange", opacity=0.7),
                        name=r"$\mathcal{C}^q \text{ Boundary Points}$",
                        legendgroup="boundary",
                        showlegend=(i == 0),  # Only show in legend once
                    ),
                    row=row,
                    col=col,
                )

        # Set consistent axis ranges for all subplots
        if global_all_points:
            for i in range(1, 5):  # 4 subplots
                row = ((i - 1) // 2) + 1
                col = ((i - 1) % 2) + 1

                fig.update_layout(
                    **{
                        f'scene{i if i > 1 else ""}': dict(
                            xaxis=dict(
                                title="X (m)", range=[global_mins[0], global_maxs[0]]
                            ),
                            yaxis=dict(
                                title="Y (m)", range=[global_mins[1], global_maxs[1]]
                            ),
                            zaxis=dict(
                                title="θ (rad)", range=[global_mins[2], global_maxs[2]]
                            ),
                            aspectmode="cube",
                            camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)),
                        )
                    }
                )

        # Adjust subplot titles to be closer to the plots
        for annotation in fig.layout.annotations:
            annotation.y -= 0.025

        # Save interactive HTML file
        html_filename = f"fig_3D_{base_id}_interactive.html"
        html_path = plots_3d_dir / html_filename
        fig.write_html(str(html_path))
        print(f"Saved interactive 3D plot: {html_filename}")

        # Generate 360 static images for animation with proper file handle management
        print("Generating 360 static images for animation...")
        azimuth_angles = list(range(0, 360, 1))

        import errno
        import math
        import os

        try:
            import importlib

            importlib.import_module("kaleido")
            static_frames_enabled = True
        except ImportError:
            print(
                "kaleido not installed -> skipping static PNG/PDF/SVG frames (interactive HTML still generated)."
            )
            static_frames_enabled = False

        # Try to import psutil for file descriptor monitoring
        try:
            import psutil

            process = psutil.Process()
            initial_fds = process.num_fds()
            print(f"Initial file descriptors: {initial_fds}")
            monitor_fds = True
        except ImportError:
            print("psutil not available - file descriptor monitoring disabled")
            monitor_fds = False
            initial_fds = 0

        frames_generated = 0
        # Generate frames with different azimuth angles
        for az_val in azimuth_angles if static_frames_enabled else []:
            try:
                # Convert azimuth angle to camera position
                az_rad = math.radians(az_val)
                elev_rad = math.radians(20)  # Fixed elevation

                # Convert to Cartesian coordinates for camera position
                camera_x = 1.5 * math.cos(elev_rad) * math.cos(az_rad)
                camera_y = 1.5 * math.cos(elev_rad) * math.sin(az_rad)
                camera_z = 1.5 * math.sin(elev_rad)

                # Update camera position for all subplots
                fig.update_layout(
                    scene=dict(
                        camera=dict(eye=dict(x=camera_x, y=camera_y, z=camera_z))
                    ),
                    scene2=dict(
                        camera=dict(eye=dict(x=camera_x, y=camera_y, z=camera_z))
                    ),
                    scene3=dict(
                        camera=dict(eye=dict(x=camera_x, y=camera_y, z=camera_z))
                    ),
                    scene4=dict(
                        camera=dict(eye=dict(x=camera_x, y=camera_y, z=camera_z))
                    ),
                )

                frame_base_path = plots_3d_dir / f"fig_3D_{base_id}_az{az_val:03d}"
                for extension in ("png", "pdf", "svg"):
                    frame_path = frame_base_path.with_suffix(f".{extension}")
                    fig.write_image(str(frame_path), width=1400, height=1200, scale=2)
                frames_generated += 1

                if az_val % 30 == 0:  # Print progress every 30 degrees
                    if monitor_fds:
                        current_fds = process.num_fds()
                        print(
                            f"  Generated frame at {az_val}° - FDs: {current_fds} (Δ: {current_fds - initial_fds})"
                        )
                    else:
                        print(f"  Generated frame at {az_val}°")

                    # Aggressive cleanup every 30 frames
                    gc.collect()
                    time.sleep(0.1)  # Brief pause to allow cleanup

            except OSError as e:
                if e.errno == errno.EMFILE:  # Too many open files
                    print(
                        f"File handle limit reached at frame {az_val}. Generated {frames_generated} frames."
                    )
                    gc.collect()
                    time.sleep(1.0)  # Pause to release handles
                    break
                else:
                    raise
            except Exception as e:
                if az_val == 0:  # Only print error once
                    print(f"Note: Could not save static PNG/PDF/SVG frames: {e}")
                break

        print(f"Generated {frames_generated} animation frames")
        if monitor_fds:
            final_fds = process.num_fds()
            print(f"Final file descriptors: {final_fds} (Δ: {final_fds - initial_fds})")

        # Clean up the original figure object to release memory
        del fig
        gc.collect()

        # Free memory after processing this validation ID
        del raw_data, full_mc_particles, mc_particles_mean, method_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force cleanup of any remaining file handles
        gc.collect()
        import ctypes

        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.sync()  # Force filesystem sync
        except:
            pass  # Ignore if libc not available

        print(f"  Memory freed for {base_id}")

    print(f"✅ 3D plots saved to: {plots_3d_dir}")


def generate_latex_table_core(
    table_data, failure_rate=0.10, n_validation=None, robot_name="Isaac_Jetbot"
):
    """Core LaTeX table generation logic (extracted for reuse)."""
    # The table_data now contains pre-calculated ratios
    target_coverage = (1 - failure_rate) * 100

    # Generate LaTeX table with colortbl support
    latex_table = r"""\begin{table}[h]
\vspace{-3mm}
\captionsetup{              % only affects *this* table
  skip   = 4pt,             % vertical gap ↓ between caption and table
}
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
% \renewcommand{\arraystretch}{1}
"""

    # Check if we need transposed table for Real_MBot
    if robot_name == "Real_MBot":
        # Extract CLAPS volume and area from table_data
        claps_volume = 1.0  # CLAPS volume ratio is always 1.0 (reference)
        claps_area = 1.0  # CLAPS area ratio is always 1.0 (reference)

        # Find CLAPS data if we need actual values
        for data in table_data:
            if "CLAPS" in data["method"]:
                # For now, we use the ratios which are 1.0 for CLAPS
                # If we need actual values, they would need to be passed separately
                break

        return generate_transposed_table(
            table_data,
            failure_rate,
            n_validation,
            target_coverage,
            claps_volume,
            claps_area,
        )

    # Add validation sample size to caption if available
    if n_validation:
        latex_table += f"{{\\setstretch{{0.8}}\\caption{{Marginal Coverage and Volume Ratio (over ${n_validation}$ validation trials)}}}}\n"
    else:
        latex_table += (
            r"{\setstretch{0.8}\caption{Marginal Coverage and Volume Ratio}}" + "\n"
        )

    latex_table += r"""\vspace{-3mm}
\label{tab:isaac_results}
\begin{tabular}{l|c|c|c|c}
\toprule
Algorithm & Marginal Coverage & Volume Ratio & 2D Area Ratio & IoU vs MC\\
& $(\%)$ $\uparrow$ & (relative to CLAPS) $\downarrow$ & (relative to CLAPS) $\downarrow$ & $(\%)$ $\uparrow$\\
\midrule
"""

    # Add each method row
    for i, data in enumerate(table_data):
        method = data["method"]  # This already contains the full display name
        coverage = data["coverage"]
        volume_ratio = data.get("volume_ratio", 0.0)
        area_ratio = data.get("area_ratio", 0.0)
        iou_vs_mc = data.get("iou_vs_mc", 0.0)

        # Determine row color based on coverage achievement
        if coverage >= target_coverage:
            row_color = r"\rowcolor{green!15}"
        else:
            row_color = r"\rowcolor{red!15}"

        # Simple coverage value (no bold/underline/checkmarks)
        coverage_value = f"{coverage:.2f}"

        # Determine formatting for volume ratio
        if volume_ratio == 1.0:  # This will be CLAPS
            ratio_str = r"\textbf{" + f"{volume_ratio:.2f}" + r"}"
        elif (
            volume_ratio
            == sorted(
                [
                    d.get("volume_ratio", 0)
                    for d in table_data
                    if d.get("volume_ratio", 0) > 0
                ]
            )[0]
        ):  # Smallest ratio (best)
            ratio_str = r"\underline{" + f"{volume_ratio:.2f}" + r"}"
        else:
            ratio_str = f"{volume_ratio:.2f}"

        # Format 2D area ratio
        if area_ratio > 0:
            # Determine formatting for area ratio
            if area_ratio == 1.0:  # This will be CLAPS
                area_str = r"\textbf{" + f"{area_ratio:.2f}" + r"}"
            elif (
                area_ratio
                == sorted(
                    [
                        d.get("area_ratio", 0)
                        for d in table_data
                        if d.get("area_ratio", 0) > 0
                    ]
                )[0]
            ):  # Smallest ratio (best)
                area_str = r"\underline{" + f"{area_ratio:.2f}" + r"}"
            else:
                area_str = f"{area_ratio:.2f}"
        else:
            area_str = "N/A"

        # Format IoU
        iou_str = f"{iou_vs_mc * 100:.1f}" if iou_vs_mc > 0 else "N/A"

        # Add row with full row color
        latex_table += f"{row_color} {method} & {coverage_value} & {ratio_str} & {area_str} & {iou_str}\\\\ \n"
        if i < len(table_data) - 1:
            latex_table += r"\midrule" + "\n"

    # Close table
    latex_table += r"""\end{tabular}
\noindent\begin{minipage}{\linewidth}
  \scriptsize\raggedright
Row coloring: \colorbox{green!15}{Light green} indicates coverage $\geq$ target $(1-\targetalpha)$; \colorbox{red!15}{Light red} indicates coverage below target.\\
Volume Ratio: prediction region volume was divided by the \textit{smallest}'s region volume.
\end{minipage}
\vspace{-6mm}
\end{table}"""

    return latex_table


def generate_transposed_table(
    table_data, failure_rate, n_validation, target_coverage, claps_volume, claps_area
):
    """Generate transposed table for Real_MBot (columns are algorithms, rows are metrics)."""

    # Prepare data for transposition - abbreviate method names for compact table
    methods = []
    for data in table_data:
        method = data["method"]
        # Abbreviate method names for compact transposed layout
        if "Baseline" in method:
            abbrev = method.replace("Baseline ", "B")
        else:
            abbrev = method
        methods.append(abbrev)

    coverages = [data["coverage"] for data in table_data]
    volume_ratios = [data.get("volume_ratio", 0.0) for data in table_data]

    # Start with complete table wrapper
    latex_table = r"""\begin{table}[h]
\vspace{-3mm}
\captionsetup{              % only affects *this* table
  skip   = 4pt,             % vertical gap ↓ between caption and table
}
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
% \renewcommand{\arraystretch}{1}
"""

    # Add validation sample size to caption if available
    if n_validation:
        latex_table += f"{{\\setstretch{{0.8}}\\caption{{Marginal Coverage and Volume Ratio (over ${n_validation}$ validation trials)}}}}\n"
    else:
        latex_table += (
            r"{\setstretch{0.8}\caption{Marginal Coverage and Volume Ratio}}" + "\n"
        )

    latex_table += r"""\vspace{-3mm}
\label{tab:mbot_results}
"""

    # Create column specification dynamically - one column per method
    col_spec = "l|" + "c|" * len(methods)
    latex_table += f"\\begin{{tabular}}{{{col_spec[:-1]}}}\n"  # Remove last |
    latex_table += "\\toprule\n"

    # Header row - method names
    header_row = "Metric & " + " & ".join(methods) + "\\\\\n"
    latex_table += header_row
    latex_table += "\\midrule\n"

    # Row 1: Coverage values
    coverage_values = []
    for i, coverage in enumerate(coverages):
        if coverage >= target_coverage:
            coverage_values.append(f"\\textbf{{{coverage:.1f}}}")
        else:
            coverage_values.append(f"{{{coverage:.1f}}}")

    coverage_row = (
        f"Coverage (\\%) $\\uparrow$ & " + " & ".join(coverage_values) + "\\\\\n"
    )
    latex_table += coverage_row

    # Row 2: Volume ratios
    ratio_values = []
    for i, ratio in enumerate(volume_ratios):
        if ratio == 1.0:  # CLAPS reference
            ratio_values.append(f"\\textbf{{{ratio:.2f}}}")
        elif ratio == min(volume_ratios):  # Best (smallest) ratio
            ratio_values.append(f"\\underline{{{ratio:.2f}}}")
        else:
            ratio_values.append(f"{ratio:.2f}")

    ratio_row = (
        f"Volume Ratio (relative to CLAPS) $\\downarrow$ & "
        + " & ".join(ratio_values)
        + "\\\\\n"
    )
    latex_table += ratio_row

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}\n"

    return latex_table


def process_method_file_complete(args):
    """Process ONE validation file for ONE method - complete pipeline.

    Does everything for a single file:
    1. Load metrics data
    2. Fit polygon to boundary points
    3. Compute IoU with MC polygon
    4. Generate debug plots if IoU is suspicious

    This is self-contained - each worker handles complete processing.

    Args:
        args: Tuple containing (metrics_file_path, method_name, run_id, robot_radius, mc_polygon_dict)

    Returns:
        dict: Complete result with data, polygon, IoU, etc.
    """
    # Import required modules for spawn compatibility
    import gc
    import time
    from pathlib import Path

    import alphashape
    import cv2
    import numpy as np
    import torch
    from scipy.spatial import ConvexHull
    from shapely.geometry import Polygon

    metrics_file_path, method_name, run_id, mc_polygon_dict = args

    def fit_polygon_to_points_worker(points_2d, method_name="unknown", run_id=None):
        """Worker version of fit_polygon_to_points - self-contained."""
        # Convert tensor to numpy if needed for consistent processing
        if hasattr(points_2d, "cpu"):  # Check if it's a PyTorch tensor
            points_np = points_2d.cpu().numpy()
        else:
            points_np = points_2d

        if len(points_np) < 3:
            return None, 0.0, None, {}

        original_count = len(points_np)

        # Methods that are known to be convex - use fast ConvexHull path
        convex_methods = ["BASELINE1", "BASELINE2", "BASELINE3", "BASELINE5"]
        method_upper = method_name.upper()
        use_convex_hull = any(m in method_upper for m in convex_methods)

        if use_convex_hull:
            # FAST PATH: Direct ConvexHull for convex methods
            start_time = time.time()

            hull = ConvexHull(points_np)
            hull_points = points_np[hull.vertices]

            # Create polygon from hull
            polygon = Polygon(hull_points)
            coords = np.vstack([hull_points, hull_points[0]])  # Close the polygon
            area = polygon.area

            approach_name = "ConvexHull"
            processing_points = len(hull_points)
            elapsed = time.time() - start_time

            # Store debug data for later plotting
            debug_data = {
                "original_points": points_np,
                "boundary_points": hull_points,
                "approach_name": approach_name,
            }

        else:
            # SLOWER PATH: Boundary extraction + alphashape for concave methods
            start_time = time.time()
            approach_name = "CV2+AlphaShape"

            # Store original points for debug plotting before boundary extraction
            original_points_for_debug = points_np.copy()

            # Extract boundary points - inline the logic to avoid function call
            extract_start = time.time()

            boundary_fraction = get_boundary_fraction(method_name)

            # Inline extract_boundary_points logic

            if len(points_np) < 3:
                boundary_pts = points_np
            else:
                # Use ConvexHull to get boundary
                hull = ConvexHull(points_np)
                convex_boundary = points_np[hull.vertices]

                # Keep a fraction of all points for non-convex shapes
                n_to_keep = max(10, int(boundary_fraction * len(points_np)))
                if len(points_np) <= n_to_keep:
                    boundary_pts = points_np
                else:
                    # Random sampling + convex hull
                    np.random.seed(42)
                    random_indices = np.random.choice(
                        len(points_np), n_to_keep, replace=False
                    )
                    random_points = points_np[random_indices]

                    # Combine convex hull + random sample
                    all_boundary = np.vstack([convex_boundary, random_points])

                    # Remove duplicates
                    unique_boundary = np.unique(all_boundary, axis=0)
                    boundary_pts = unique_boundary

            extract_time = time.time() - extract_start

            extract_time = time.time() - extract_start

            points_np = boundary_pts

            # Use alphashape with hardcoded alpha for non-convex shapes (fast, no optimization)
            poly_start = time.time()

            # Memory optimization: Ensure points are contiguous and cleaned up
            points_np = np.ascontiguousarray(points_np)

            # Use hardcoded alpha for speed while preserving non-convex shapes
            alpha = 100  # Adjust this value as needed for tightness
            polygon = alphashape.alphashape(points_np, alpha)

            # Explicit memory cleanup after alphashape creation
            gc.collect()

            poly_time = time.time() - poly_start

            # Get coordinates for plotting (ensure polygon is closed)
            if hasattr(polygon, "exterior"):
                coords = np.array(polygon.exterior.coords)
                area = polygon.area
            else:
                # Handle case where hull is a line or point
                coords = np.array([[0, 0], [0, 0], [0, 0]])  # Dummy triangle
                area = 0.0

            total_time = time.time() - start_time

            # Store debug data for later plotting
            debug_data = {
                "original_points": original_points_for_debug,
                "boundary_points": points_np,
                "approach_name": approach_name,
                "boundary_fraction": boundary_fraction,
            }

        return polygon, area, coords, debug_data

    def compute_polygon_iou_worker(polygon1, polygon2):
        """Worker version of compute_polygon_iou - self-contained."""
        if polygon1 is None or polygon2 is None:
            raise ValueError("Polygon is None")

        # Compute intersection and union
        intersection = polygon1.intersection(polygon2)
        union = polygon1.union(polygon2)

        # Handle MultiPolygon results
        if hasattr(intersection, "geoms"):  # MultiPolygon
            intersection_area = sum(p.area for p in intersection.geoms)
        else:
            intersection_area = intersection.area

        if hasattr(union, "geoms"):  # MultiPolygon
            union_area = sum(p.area for p in union.geoms)
        else:
            union_area = union.area

        # Compute IoU
        if union_area > 0:
            iou = intersection_area / union_area
        else:
            raise ValueError("Union area is 0")

        return iou

    def tensordict_to_native(d):
        """Convert a dict with torch tensors to native Python types for pickling."""
        result = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                # Convert scalar tensors to Python floats/ints
                if v.numel() == 1:
                    result[k] = v.item()
                else:
                    # Convert array tensors to numpy (pickle-safe)
                    result[k] = v.cpu().numpy()
            elif isinstance(v, dict):
                result[k] = tensordict_to_native(v)
            elif isinstance(v, list):
                result[k] = [
                    x.item() if isinstance(x, torch.Tensor) and x.numel() == 1
                    else (x.cpu().numpy() if isinstance(x, torch.Tensor) else x)
                    for x in v
                ]
            else:
                result[k] = v
        return result

    try:
        # Load metrics data
        data = torch.load(metrics_file_path, map_location="cpu", weights_only=False)

        # Initialize result structure
        result = {
            "run_id": run_id,
            "method_name": method_name,
            "data": None,  # Will be set after conversion
            "polygon": None,
            "area": 0.0,
            "coords": None,
            "iou": 0.0,
            "success": False,
        }

        # Extract boundary points and fit polygon
        if "boundary_SS_pts" in data and data["boundary_SS_pts"] is not None:
            boundary_pts = data["boundary_SS_pts"]

            if (
                hasattr(boundary_pts, "ndim")
                and boundary_pts.ndim >= 2
                and boundary_pts.shape[1] >= 2
            ):
                # Extract 2D coordinates
                points_2d = boundary_pts[:, :2]

                # Fit polygon using worker version
                polygon, area, coords, debug_data = fit_polygon_to_points_worker(
                    points_2d, method_name=method_name, run_id=run_id
                )
                result["debug_data"] = debug_data

                if polygon is not None:
                    result["polygon"] = polygon
                    result["area"] = area
                    result["coords"] = coords
                    result["success"] = True

                    # Compute IoU if MC polygon is available
                    if mc_polygon_dict and mc_polygon_dict.get("polygon") is not None:
                        mc_polygon = mc_polygon_dict["polygon"]

                        # Fix invalid polygons before IoU computation
                        fixed_polygon = polygon
                        fixed_mc_polygon = mc_polygon

                        if not polygon.is_valid:
                            fixed_polygon = polygon.buffer(0)
                        if not mc_polygon.is_valid:
                            fixed_mc_polygon = mc_polygon.buffer(0)

                        # Compute IoU using worker version
                        iou = compute_polygon_iou_worker(
                            fixed_polygon, fixed_mc_polygon
                        )
                        result["iou"] = iou

        # Convert data dict to native Python types for safe pickling across processes
        result["data"] = tensordict_to_native(data)

        # Clean up memory and file handles
        del data
        gc.collect()

        return result

    except Exception as e:
        # Clean up memory even on error
        gc.collect()

        return {
            "run_id": run_id,
            "method_name": method_name,
            "error": str(e),
            "success": False,
        }


def process_single_mc_polygon(args):
    """Process a single validation file for MC polygon fitting.

    This function is designed to run in a separate process for parallel execution.

    Args:
        args: Tuple containing (val_file_path)

    Returns:
        tuple: (run_id, polygon_data_dict)
    """
    # Import required modules for spawn compatibility
    import matplotlib
    import numpy as np
    import torch

    matplotlib.use("Agg")
    import gc
    from pathlib import Path

    import matplotlib.pyplot as plt

    val_file_path = args

    # Load file
    raw_data = torch.load(val_file_path, map_location="cpu", weights_only=False)
    mc_particles = raw_data["q1"]  # Shape: [100000, 3]

    # Extract 2D coordinates and fit polygon
    mc_points_2d = mc_particles[:, :2]
    run_id = Path(val_file_path).stem
    polygon, area, coords, debug_data = fit_polygon_to_points(
        mc_points_2d, method_name="MC_particles", run_id=run_id
    )

    # Clean up memory
    del raw_data, mc_particles
    import gc

    gc.collect()
    plt.close("all")

    return run_id, {
        "polygon": polygon,
        "area": area,
        "coords": coords,
        "debug_data": debug_data,
    }


def compute_mc_polygons_phase(
    robot_name, confidence_level, boundary_points, logger, dry_run=False
):
    """Phase 1: Compute MC polygons and save to disk."""
    try:
        # Use a single ProcessPoolExecutor for MC polygon processing
        n_workers = min(MAX_WORKERS, cpu_count())
        logger.info(f"🚀 Using {n_workers} workers for MC polygon processing")

        # Load MC polygon data in parallel for speed
        mc_polygons = {}
        validation_dir = Path(f"data/{robot_name}/raw_data/validation")

        if not validation_dir.exists():
            logger.error(f"Validation directory not found: {validation_dir}")
            return False

        val_files = sorted(validation_dir.glob("*.pt"))

        # Limit files for dry-run mode
        if dry_run:
            val_files = val_files[:5]  # Only process first 5 files in dry-run
            logger.info(
                f"🔬 DRY-RUN MODE: Processing {len(val_files)} MC particle files..."
            )
        else:
            logger.info(
                f"🔶 Processing {len(val_files)} MC particle files for polygon fitting..."
            )

        # Process in parallel using multiprocessing.Pool for better stability
        # Recycle workers after 50 tasks to prevent memory leaks without excessive overhead.
        with Pool(processes=n_workers, maxtasksperchild=50) as pool:
            work_items = [str(val_file) for val_file in val_files]
            
            # Use imap_unordered for iterator-based processing
            results_iter = pool.imap_unordered(process_single_mc_polygon, work_items)

            # Process results as they complete
            for run_id, polygon_data in tqdm(
                results_iter,
                total=len(work_items),
                desc="Fitting MC polygons (parallel)",
                unit="files",
            ):
                try:
                    if run_id and polygon_data:
                        mc_polygons[run_id] = polygon_data
                        logger.debug(f"  MC {run_id}: area = {polygon_data['area']:.6f}")
                except Exception as e:
                    logger.warning(f"Failed to process MC polygon result: {e}")

        logger.info(f"🔶 Fitted polygons for {len(mc_polygons)} MC particle sets")

        # Save MC polygons to disk
        reports_dir = Path(
            f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}"
        )
        reports_dir.mkdir(parents=True, exist_ok=True)
        mc_polygons_file = reports_dir / "mc_polygons.pt"

        torch.save(mc_polygons, mc_polygons_file)
        logger.info(f"💾 Saved MC polygons to: {mc_polygons_file}")

        return True

    except Exception as e:
        logger.error(f"❌ MC polygon phase failed: {e}")
        return False


def load_mc_polygons(robot_name, confidence_level, boundary_points, logger):
    """Load MC polygons from disk."""
    reports_dir = Path(
        f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}"
    )
    mc_polygons_file = reports_dir / "mc_polygons.pt"

    if not mc_polygons_file.exists():
        logger.error(f"❌ MC polygons file not found: {mc_polygons_file}")
        logger.error("   Run Phase 1 first: --mc-only")
        return None

    try:
        mc_polygons = torch.load(
            mc_polygons_file, map_location="cpu", weights_only=False
        )
        logger.info(
            f"📂 Loaded {len(mc_polygons)} MC polygons from: {mc_polygons_file}"
        )
        return mc_polygons
    except Exception as e:
        logger.error(f"❌ Failed to load MC polygons: {e}")
        return None


def process_methods_phase(
    robot_name,
    confidence_level,
    boundary_points,
    logger,
    recompute=True,
    dry_run=False,
    parallel_methods=False,
    method_workers=None,
):
    """Phase 2: Process methods using saved MC polygons, compute all metrics."""
    try:
        # Load MC polygons when we need to recompute polygon/IoU metrics
        mc_polygons = {}
        if recompute:
            mc_polygons = load_mc_polygons(
                robot_name, confidence_level, boundary_points, logger
            )
            if mc_polygons is None:
                return False
        else:
            logger.info(
                "⏩ Skip recompute enabled: using polygons saved in validation metrics"
            )
            optional_mc_file = Path(
                f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}/mc_polygons.pt"
            )
            if optional_mc_file.exists():
                try:
                    mc_polygons = torch.load(
                        optional_mc_file, map_location="cpu", weights_only=False
                    )
                    logger.info(
                        f"📂 Loaded optional MC polygons from: {optional_mc_file}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"⚠️  Could not load optional MC polygons ({optional_mc_file}): {exc}"
                    )

        # Load existing processed data from new structure
        experiment_dir = Path(
            f"data/{robot_name}/experiments/{confidence_level}/boundary_{boundary_points}"
        )

        if not experiment_dir.exists():
            logger.error(f"No experiment directory found: {experiment_dir}")
            return False

        # Process each method completely before moving to next (minimal memory usage)
        method_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]

        # Sort to ensure CLAPS is processed first (needed for volume ratios)
        method_dirs_sorted = sorted(
            method_dirs, key=lambda d: (d.name != "CLAPS", d.name)
        )

        # Only keep what we actually need
        method_summaries = {}
        claps_volumes = None  # Only raw data we keep
        claps_ious = None  # Keep IoU data for ratio computation
        volume_ratios = {}
        iou_ratios = {}
        motion_data = {
            "dq0": [],
            "dq1": [],
            "u": [],
        }  # Store motion parameters for correlation analysis
        polygon_data = {}  # Store polygon coordinates for each method and run
        n_validation_runs = 0

        # Track file counts per method for verification
        method_file_counts = {}

        # Add MC polygons to polygon_data for saving
        polygon_data["MC_particles"] = mc_polygons

        logger.info(
            f"🚀 Processing methods sequentially to avoid multiprocessing issues"
        )

        # Process each method; optionally parallelize per-file work
        for method_dir in tqdm(method_dirs_sorted, desc="Processing methods"):
            method_name = method_dir.name
            validation_metrics_dir = method_dir / "validation" / "metrics"

            if not validation_metrics_dir.exists():
                continue

            # Gather all files for this method
            metrics_files = list(sorted(validation_metrics_dir.glob("*.pt")))

            # Limit files for dry-run mode
            if dry_run:
                original_count = len(metrics_files)
                metrics_files = metrics_files[
                    :5
                ]  # Only process first 5 files in dry-run
                logger.info(
                    f"🔬 DRY-RUN MODE: Processing {len(metrics_files)}/{original_count} files for {method_name}"
                )

            method_file_counts[method_name] = {
                "found": len(metrics_files),
                "loaded": 0,
                "failed": 0,
            }

            if not metrics_files:
                logger.warning(f"⚠️ No files found for method {method_name}")
                continue

            # Process files sequentially to avoid multiprocessing issues
            method_data = []
            polygon_data[method_name] = {}

            if not recompute:
                logger.info(
                    f"⏩ Using cached metrics for {method_name} (skip polygon refit)"
                )
                for metrics_file in tqdm(
                    metrics_files, desc=f"  {method_name} files", leave=False
                ):
                    try:
                        run_id = metrics_file.stem.replace(f"_{method_name}", "")
                        data = torch.load(
                            metrics_file, map_location="cpu", weights_only=False
                        )

                        method_data.append(data)
                        method_file_counts[method_name]["loaded"] += 1

                        # Keep lightweight polygon metadata if available
                        if (
                            "polygon_coords" in data
                            or "area_2d" in data
                            or "iou_vs_mc" in data
                        ):
                            polygon_data[method_name][run_id] = {
                                "coords": data.get("polygon_coords"),
                                "area": data.get("area_2d", 0.0),
                                "iou_vs_mc": data.get("iou_vs_mc", 0.0),
                            }

                    except Exception as e:
                        method_file_counts[method_name]["failed"] += 1
                        logger.error(f"Failed to load {metrics_file}: {e}")
                # Ensure we keep n_validation_runs in sync even in skip path
                n_validation_runs = max(n_validation_runs, len(method_data))

            elif parallel_methods:
                n_workers = method_workers or MAX_WORKERS
                n_workers = min(n_workers, cpu_count())
                logger.info(
                    f"🚀 Parallel mode: {len(metrics_files)} files for {method_name} using {n_workers} workers"
                )
                work_items = []
                ordered_indices = {}
                for idx, metrics_file in enumerate(metrics_files):
                    run_id = metrics_file.stem.replace(f"_{method_name}", "")
                    mc_polygon_data = mc_polygons.get(run_id, {})
                    
                    # Create lightweight version for worker to avoid pickling heavy debug_data (100k points)
                    lightweight_mc_data = {
                        "polygon": mc_polygon_data.get("polygon"),
                        "area": mc_polygon_data.get("area"),
                    }
                    
                    work_items.append(
                        (str(metrics_file), method_name, run_id, lightweight_mc_data)
                    )
                    ordered_indices[run_id] = idx

                method_data_buffer = [None] * len(metrics_files)
                
                # NOTE: We explicitly set max_workers=1 for certain robot types or configurations
                # if they are known to be unstable with multiprocessing.
                # However, the error "A process in the process pool was terminated abruptly"
                # usually indicates OOM (Out Of Memory) or a segfault in a worker.
                # We can try to reduce workers if we detect issues, or just use safe defaults.
                
                effective_workers = n_workers
                # If we are running locally and face OOM, we might want to cap this
                # if effective_workers > 8: 
                #     effective_workers = 8

                try:

                    # Use multiprocessing.Pool with maxtasksperchild to prevent memory leaks
                    # NOTE: maxtasksperchild=1 is too slow (respawns process after EVERY file).
                    # Using 50 gives good balance between memory management and performance.
                    with Pool(processes=effective_workers, maxtasksperchild=50) as pool:
                        # Use imap_unordered to get results as they finish
                        results_iter = pool.imap_unordered(process_method_file_complete, work_items)
                        
                        for result in tqdm(
                            results_iter,
                            total=len(work_items),
                            desc=f"  {method_name} files (parallel)",
                            leave=False,
                        ):
                            try:
                                # Find the original index for this result to maintain order in buffer if needed
                                # But actually we just need to store it.
                                # Since we used imap_unordered, we don't know the index directly.
                                # But we can look it up if we really need to, or just append to a list.
                                # However, the original code put it in method_data_buffer[idx].
                                # Let's just collect valid results.
                                
                                if result.get("success", False):
                                    data = result["data"]
                                    data["polygon_2d"] = result["polygon"]
                                    data["area_2d"] = result["area"]
                                    data["polygon_coords"] = result["coords"]
                                    data["iou_vs_mc"] = result["iou"]

                                    # We can't easily put it in the buffer at the exact index without passing index through.
                                    # But the buffer was mainly used to collect results.
                                    # Let's just append to a list and sort later if order matters (it usually doesn't for metrics aggregation).
                                    # Actually, let's just add to method_data directly.
                                    
                                    # Wait, the original code used method_data_buffer to keep them in order?
                                    # The original code: method_data_buffer[idx] = data
                                    # And then: method_data = [d for d in method_data_buffer if d is not None]
                                    # So order might have been preserved but filtered.
                                    # Let's just add to a list.
                                    
                                    method_data.append(data)
                                    method_file_counts[method_name]["loaded"] += 1

                                    if result["polygon"] is not None:
                                        polygon_data[method_name][result["run_id"]] = {
                                            "polygon": result["polygon"],
                                            "area": result["area"],
                                            "coords": result["coords"],
                                            "iou_vs_mc": result["iou"],
                                            "debug_data": result.get("debug_data"),
                                        }

                                    logger.debug(
                                        f"  {result['run_id']}: {method_name} area = {result['area']:.6f}, IoU = {result['iou']:.4f}"
                                    )
                                else:
                                    method_file_counts[method_name]["failed"] += 1
                                    if "error" in result:
                                        logger.warning(
                                            f"Failed to process {result.get('run_id', 'unknown')}: {result['error']}"
                                        )

                            except Exception as e:
                                method_file_counts[method_name]["failed"] += 1
                                logger.error(f"Failed to process result: {e}")
                                continue

                except Exception as e:
                     logger.error(f"💥 Unexpected error during parallel processing {method_name}: {e}")

            else:
                logger.info(
                    f"🚀 Processing {len(metrics_files)} files for {method_name} sequentially"
                )

                for metrics_file in tqdm(
                    metrics_files, desc=f"  {method_name} files", leave=False
                ):
                    try:
                        run_id = metrics_file.stem.replace(f"_{method_name}", "")
                        mc_polygon_data = mc_polygons.get(run_id, {})

                        result = process_method_file_complete(
                            (str(metrics_file), method_name, run_id, mc_polygon_data)
                        )

                        if result.get("success", False):
                            data = result["data"]
                            data["polygon_2d"] = result["polygon"]
                            data["area_2d"] = result["area"]
                            data["polygon_coords"] = result["coords"]
                            data["iou_vs_mc"] = result["iou"]

                            method_data.append(data)
                            method_file_counts[method_name]["loaded"] += 1

                            if result["polygon"] is not None:
                                polygon_data[method_name][result["run_id"]] = {
                                    "polygon": result["polygon"],
                                    "area": result["area"],
                                    "coords": result["coords"],
                                    "iou_vs_mc": result["iou"],
                                    "debug_data": result.get("debug_data"),
                                }

                            logger.debug(
                                f"  {result['run_id']}: {method_name} area = {result['area']:.6f}, IoU = {result['iou']:.4f}"
                            )
                        else:
                            method_file_counts[method_name]["failed"] += 1
                            if "error" in result:
                                logger.warning(
                                    f"Failed to process {result.get('run_id', 'unknown')}: {result['error']}"
                                )
                            dummy_data = {
                                "polygon_2d": None,
                                "area_2d": 0.0,
                                "polygon_coords": None,
                                "iou_vs_mc": 0.0,
                            }
                            method_data.append(dummy_data)

                    except Exception as e:
                        method_file_counts[method_name]["failed"] += 1
                        logger.error(f"Failed to process {metrics_file}: {e}")

            if not method_data:
                logger.warning(
                    f"⚠️ No data processed successfully for method {method_name}"
                )
                continue

            # Track validation runs count
            n_validation_runs = max(n_validation_runs, len(method_data))

            logger.info(
                f"✅ Completed {method_name}: {method_file_counts[method_name]['loaded']} loaded, {method_file_counts[method_name]['failed']} failed"
            )

            # Compute summary immediately
            method_summary = _compute_method_summary(method_data)
            method_summaries[method_name] = method_summary

            # Special handling for CLAPS - keep volumes and extract motion data
            if method_name == "CLAPS":
                claps_volumes = method_summary["raw_values"].get(
                    "total_volume_mesh", []
                )
                if not claps_volumes:
                    raise ValueError(
                        "CLAPS method found but has no volume data - this should not happen"
                    )

                claps_ious = method_summary["raw_values"].get("iou_vs_mc", [])
                if not claps_ious:
                    logger.warning("CLAPS method found but has no IoU data")
                    claps_ious = []

                # Extract motion data for correlation analysis (indexed by validation run)
                for data in method_data:
                    if "dq0" in data and "dq1" in data and "u" in data:
                        # Extract velocity and acceleration data
                        dq0 = data["dq0"]
                        dq1 = data["dq1"]
                        u = data["u"]
                        motion_data["dq0"].append(dq0)
                        motion_data["dq1"].append(dq1)
                        motion_data["u"].append(u)

            # For BASELINE methods, compute volume ratio relative to CLAPS
            if method_name.startswith("BASELINE"):
                if claps_volumes is None:
                    raise ValueError(
                        "BASELINE method found before CLAPS - CLAPS must be processed first"
                    )

                baseline_volumes = method_summary["raw_values"].get(
                    "total_volume_mesh", []
                )
                if baseline_volumes:
                    # Match by index (same validation cases)
                    min_length = min(len(claps_volumes), len(baseline_volumes))
                    ratios = []
                    for i in range(min_length):
                        if claps_volumes[i] > 0:
                            ratio = baseline_volumes[i] / claps_volumes[i]
                            ratios.append(ratio)

                    if ratios:
                        volume_ratios[f"{method_name}_vs_CLAPS"] = ratios

                    # Compute IoU ratios for individual runs
                    baseline_ious = method_summary["raw_values"].get("iou_vs_mc", [])
                    if baseline_ious and claps_ious:
                        iou_ratios_list = []
                        min_length = min(len(baseline_ious), len(claps_ious))
                        for i in range(min_length):
                            if claps_ious[i] > 0:
                                ratio = baseline_ious[i] / claps_ious[i]
                                iou_ratios_list.append(ratio)

                        if iou_ratios_list:
                            iou_ratios[f"{method_name}_vs_CLAPS"] = iou_ratios_list

            del method_data
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Brief pause to allow system cleanup
            import time

            time.sleep(0.5)

        if not method_summaries:
            logger.error("No metrics files found")
            return False

        # Save results to disk
        logger.info("💾 Saving method results to disk...")
        reports_dir = Path(
            f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}"
        )
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Save method results
        method_results = {
            "method_summaries": method_summaries,
            "polygon_data": polygon_data,
            "volume_ratios": volume_ratios,
            "motion_data": motion_data,
            "method_file_counts": method_file_counts,
            "n_validation_runs": n_validation_runs,
        }

        method_results_file = reports_dir / "method_results.pt"
        torch.save(method_results, method_results_file)
        logger.info(f"💾 Saved method results to: {method_results_file}")

        # Also save summary statistics for compatibility with plotting functions
        summary_data = {
            "metadata": {
                "robot_name": robot_name,
                "confidence_level": confidence_level,
                "boundary_points": boundary_points,
                "n_validation_runs": n_validation_runs,
                "timestamp": datetime.now().isoformat(),
                "failure_rate": 0.10,  # Default failure rate
            },
            "methods": method_summaries,
            "comparisons": {
                "volume_ratios": volume_ratios,
                "iou_ratios": iou_ratios,
                "motion_data": motion_data,
            },
            "polygon_data": polygon_data,
            "file_counts": method_file_counts,
        }

        summary_path = reports_dir / "summary_statistics.pt"
        torch.save(summary_data, summary_path)
        logger.info(f"💾 Saved summary statistics to: {summary_path}")

        return True

    except Exception as e:
        logger.error(f"❌ Method processing phase failed: {e}")
        return False


def plot_real_mbot_covariance_analysis(
    robot_name, confidence_level, boundary_points, logger
):
    """
    Plot true next points (errors) and covariance ellipses for Baseline 6 and 7.
    Specific for Real_MBot validation analysis.
    """
    if robot_name != "Real_MBot":
        return

    logger.info("🤖 Generating Real_MBot covariance analysis plots...")

    try:
        from pymatlie.se2 import SE2
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
    except ImportError as e:
        logger.error(f"Failed to import required modules for covariance analysis: {e}")
        return

    # Output directory
    reports_dir = Path(
        f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}"
    )
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Methods to analyze
    methods = ["BASELINE6", "BASELINE7"]

    fig, ax = plt.subplots(figsize=(10, 10))

    colors = {"BASELINE6": "cyan", "BASELINE7": "gray"}
    found_data = False

    def to_se2_batch(q_tensor):
        """Convert (N, 3) or (3,) pose vectors to (N, 3, 3) SE2 matrices."""
        if q_tensor.ndim == 1:
            q_tensor = q_tensor.unsqueeze(0)
        
        N = q_tensor.shape[0]
        mat = torch.eye(3, dtype=q_tensor.dtype, device=q_tensor.device).repeat(N, 1, 1)
        x = q_tensor[:, 0]
        y = q_tensor[:, 1]
        theta = q_tensor[:, 2]
        c = torch.cos(theta)
        s = torch.sin(theta)
        
        mat[:, 0, 0] = c
        mat[:, 0, 1] = -s
        mat[:, 0, 2] = x
        mat[:, 1, 0] = s
        mat[:, 1, 1] = c
        mat[:, 1, 2] = y
        return mat

    # Load errors from BASELINE6 (uncorrected errors)
    errors = []
    b6_val_dir = Path(
        f"data/{robot_name}/experiments/{confidence_level}/boundary_{boundary_points}/BASELINE6/validation/metrics"
    )
    
    if not b6_val_dir.exists():
        logger.warning(f"Validation directory not found for BASELINE6")
        return

    files = list(b6_val_dir.glob("val_real_robot_*_*.pt"))
    logger.info(f"Found {len(files)} validation files for BASELINE6")

    for val_file in files:
        try:
            val_data = torch.load(val_file, map_location="cpu", weights_only=False)
            q1 = val_data.get("q1")
            mean_pred = val_data.get("mean_pred")

            if q1 is None or mean_pred is None:
                continue

            if q1.ndim == 2 and q1.shape[1] == 3:
                 pass
            elif q1.ndim == 1:
                 q1 = q1.unsqueeze(0)

            if mean_pred.ndim == 2 and mean_pred.shape[1] == 3:
                 pass
            elif mean_pred.ndim == 1:
                 mean_pred = mean_pred.unsqueeze(0)

            q1_se2 = to_se2_batch(q1)
            mean_pred_se2 = to_se2_batch(mean_pred)
            
            # Compute error: mean_pred (uncorrected) (-) q1
            err = SE2.right_minus(mean_pred_se2, q1_se2)
            if err.shape[0] == 1:
                err = err[0]
            errors.append(err.numpy())

        except Exception as e:
            continue

    if not errors:
        logger.warning("No errors collected from BASELINE6")
        return

    errors = np.array(errors)
    
    # Plot shared errors (gray dots)
    ax.scatter(
        errors[:, 0],
        errors[:, 1],
        s=5,
        alpha=0.3,
        label="Uncorrected Errors (BASELINE6)",
        color="black",
        edgecolors="none",
    )

    # Plot Covariance Ellipses for each method
    for method in methods:
        # Load calibration
        cal_dir = Path(
            f"data/{robot_name}/experiments/{confidence_level}/boundary_{boundary_points}/{method}/calibration"
        )
        
        xi_sigma = None
        bias = None

        if method == "BASELINE6":
            cov_path = cal_dir / "xi_sigma_calibrated.pt"
            if cov_path.exists():
                data = torch.load(cov_path, map_location="cpu", weights_only=False)
                xi_sigma = data.get("xi_sigma_calibrated")
        elif method == "BASELINE7":
            mean_cov_path = cal_dir / "mean_covariance.pt"
            if mean_cov_path.exists():
                data = torch.load(mean_cov_path, map_location="cpu", weights_only=False)
                xi_sigma = data.get("xi_sigma_calibrated")
                bias = data.get("bias")

        if xi_sigma is None:
            continue

        # Center for ellipse
        center = (0, 0)
        if bias is not None:
            center = (bias[0].item(), bias[1].item())
            
        # Extract 2x2 covariance
        cov_2d = xi_sigma[:2, :2].numpy()

        # Critical value for 90% confidence in 3D projected to 2D
        critical_val_sq = chi2.ppf(0.90, df=3)
        scale_factor = np.sqrt(critical_val_sq)

        vals, vecs = np.linalg.eigh(cov_2d)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * scale_factor * np.sqrt(vals)

        ell = Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=theta,
            edgecolor=colors[method],
            facecolor="none",
            linewidth=2,
            linestyle="--" if method == "BASELINE7" else "-",
            label=f"{method} 90% Conf",
        )
        ax.add_patch(ell)

        # Plot center
        ax.plot(
            center[0],
            center[1],
            "x",
            color=colors[method],
            markersize=8,
        )

    ax.set_xlabel("Error X (m)")
    ax.set_ylabel("Error Y (m)")
    ax.set_title(
        f"Real_MBot Error Distribution & Covariance Fit\n(Body Frame, Uncorrected Errors)"
    )
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plot_path = reports_dir / "real_mbot_covariance_analysis.png"
    plt.savefig(plot_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✅ Saved Real_MBot covariance analysis to {plot_path}")


def generate_plots_phase(
    robot_name,
    confidence_level,
    boundary_points,
    plot_2d,
    plot_3d,
    plot_3d_mpl,
    debug_plots,
    logger,
    mode="all-plots",
):
    """Phase 3: Generate all plots using saved method results."""
    try:
        logger.info(f"🎨 Phase 3: Generating plots (mode={mode})...")

        # Add custom covariance analysis for Real_MBot
        if robot_name == "Real_MBot":
            plot_real_mbot_covariance_analysis(
                robot_name, confidence_level, boundary_points, logger
            )

        # Load the summary file to get the data
        reports_dir = Path(
            f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}"
        )
        summary_path = reports_dir / "summary_statistics.pt"

        if not summary_path.exists():
            logger.error(f"❌ Summary file not found: {summary_path}")
            logger.error("   Run --methods-only first to generate the summary data")
            return False

        logger.info(f"📊 Loading summary data from: {summary_path}")
        summary_data = torch.load(summary_path, map_location="cpu", weights_only=False)

        logger.info("📊 Printing summary statistics...")
        print_summary_from_file(str(summary_path), logger)

        # Paper-first: fast contour comparison, then Matplotlib 3D paper figure
        if robot_name == "Isaac_Jetbot":
            logger.info("🖼️ Generating paper contour comparison plot...")
            two_contour_comparison(robot_name, confidence_level, boundary_points, logger)

        if plot_3d_mpl and robot_name == "Isaac_Jetbot":
            logger.info("🖼️ Generating matplotlib 3D plots...")
            try:
                generate_3d_plots_from_data_paper_figure(
                    robot_name, confidence_level, boundary_points
                )
                logger.info("✅ Matplotlib 3D plots generated successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate matplotlib 3D plots: {e}")

        logger.info("🎻 Generating violin plots...")
        try:
            generate_violin_plots_from_summary(str(summary_path), logger)
            logger.info("✅ Violin plots generated successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate violin plots: {e}")

        logger.info("📊 Generating IoU ratio violin plots...")
        try:
            generate_iou_ratio_violin_plots(str(summary_path), logger)
            logger.info("✅ IoU ratio violin plots generated successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate IoU ratio violin plots: {e}")

        logger.info("📊 Generating LaTeX results table...")
        try:
            generate_latex_table_from_summary(str(summary_path), logger)
            logger.info("✅ LaTeX results table generated successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate LaTeX results table: {e}")

        if debug_plots and mode == "all-plots":
            logger.info("🔬 Generating debug plots...")
            try:
                # Generate IoU debug plots for CLAPS with low IoU
                polygon_data = summary_data.get("polygon_data", {})
                mc_polygons = polygon_data.get("MC_particles", {})
                claps_data = polygon_data.get("CLAPS", {})

                if claps_data and mc_polygons:
                    debug_plot_count = 0
                    for run_id, claps_run_data in claps_data.items():
                        if (
                            run_id in mc_polygons
                            and claps_run_data.get("iou_vs_mc", 1.0) < 0.6
                        ):
                            create_iou_debug_plot(
                                run_id,
                                "CLAPS",
                                claps_run_data["polygon"],
                                mc_polygons[run_id]["polygon"],
                                claps_run_data["iou_vs_mc"],
                            )
                            debug_plot_count += 1

                    logger.info(f"✅ Generated {debug_plot_count} IoU debug plots")

                # Generate polygon fitting debug plots for all methods
                polygon_debug_count = 0
                for method_name, method_data in polygon_data.items():
                    if method_name == "MC_particles":
                        continue
                    for run_id, run_data in method_data.items():
                        debug_data = run_data.get("debug_data")
                        if debug_data and debug_data.get("original_points") is not None:
                            plot_polygon_fitting_debug(
                                method_name,
                                run_id,
                                debug_data["original_points"],
                                debug_data["boundary_points"],
                                run_data["polygon"],
                                run_data["area"],
                                run_data["coords"],
                                debug_data["approach_name"],
                            )
                            polygon_debug_count += 1


                logger.info(
                    f"✅ Generated {polygon_debug_count} polygon fitting debug plots"
                )
                logger.info("✅ Debug plots generated successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate debug plots: {e}")

        # Generate 2D plots if requested
        if plot_2d and mode == "all-plots":
            logger.info("🖼️ Generating 2D plots...")
            try:
                generate_2d_plots_from_data(
                    robot_name,
                    confidence_level,
                    boundary_points,
                    str(summary_path),
                    dry_run=False,
                )
                logger.info("✅ 2D plots generated successfully")
                two_contour_comparison(
                    robot_name, confidence_level, boundary_points, logger
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate 2D plots: {e}")

        # Generate 3D plots if requested
        if plot_3d and mode == "all-plots":
            logger.info("🎯 Generating 3D plots...")
            try:
                generate_3d_plots_from_data(
                    robot_name, confidence_level, boundary_points
                )
                logger.info("✅ 3D plots generated successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate 3D plots: {e}")

        logger.info("🎨 Phase 3 completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Phase 3 failed: {e}")
        return False


def plot_comparison_side_by_side(
    validation_id, robot_name, confidence_level, boundary_points
):
    """Plot angle_filtered and contour plots side by side for the same validation ID.

    Args:
        validation_id: The validation run ID (e.g., "val_isaac_0007")
        robot_name: Robot name (e.g., "Isaac_Jetbot")
        confidence_level: Confidence level (e.g., "over_confident")
        boundary_points: Number of boundary points (e.g., 5000)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    def extract_xy_point(point_data, context):
        if point_data is None:
            raise ValueError(f"{context} is required for mean plotting")

        point_np = (
            point_data.cpu().numpy() if hasattr(point_data, "cpu") else np.array(point_data)
        )

        if point_np.ndim == 1 and point_np.shape[0] >= 2:
            return point_np[0], point_np[1]

        if point_np.ndim >= 2 and point_np.shape[1] >= 2:
            return point_np[0, 0], point_np[0, 1]

        raise ValueError(f"{context} must provide at least two position values")

    # Load data using helper function
    method_map, mc_particles, summary_file_path = load_validation_data_for_id(
        validation_id, robot_name, confidence_level, boundary_points
    )

    if not method_map:
        print(f"No validation data found for ID: {validation_id}")
        return

    if mc_particles is None:
        print(f"No MC particles available for {validation_id}")
        return

    # Calculate reference angle for angle filtering
    reference_angle = mc_particles[:, 2].mean().item()
    angle_threshold = 0.02

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=FIGURE_DPI)

    # Set up axes using helper function
    setup_2d_plot_axis(ax1, title="Angle Filtered")
    setup_2d_plot_axis(ax2, title="Contour")

    # Collect all points for shared axis limits
    all_points = []
    buffered_methods = ["CLAPS", "BASELINE1", "BASELINE4", "BASELINE5"]

    # Plot angle filtered (left subplot) - manual since it needs angle filtering
    for method, d in method_map.items():
        boundary = d.get("boundary_SS_pts")
        if boundary is None or boundary.ndim == 1:
            continue

        color = METHOD_COLORS.get(method, "tab:gray")
        display_name = METHOD_DISPLAY_NAMES.get(method, method)

        # Apply angle filtering
        boundary_filtered, mask = filter_data_by_angle(
            boundary, reference_angle, angle_threshold
        )
        if boundary_filtered is not None and len(boundary_filtered) > 0:
            if boundary_filtered.ndim == 1:
                boundary_filtered = boundary_filtered.reshape(1, -1)
            points_2d = boundary_filtered[:, :2]

            # Collect points for axis limits
            if method in buffered_methods:
                all_points.append(points_2d)

            # Plot scatter points
            ax1.scatter(
                points_2d[:, 0],
                points_2d[:, 1],
                s=1,
                alpha=0.6,
                color=color,
                label=display_name,
            )

    # Plot contour (right subplot) using helper function
    contour_points = plot_method_contours(
        ax2, method_map, buffered_methods, collect_points_for_limits=True
    )
    all_points.extend(contour_points)

    # Add MC particles to both plots
    if mc_particles is not None and mc_particles.shape[0] > 0:
        mc_np = mc_particles.cpu().numpy()
        all_points.append(mc_np[:, :2])

        # For angle filtered plot (ax1), filter MC particles by angle
        mc_filtered, _ = filter_data_by_angle(
            mc_particles, reference_angle, angle_threshold
        )
        if mc_filtered is not None and len(mc_filtered) > 0:
            mc_filtered_np = (
                mc_filtered.cpu().numpy()
                if hasattr(mc_filtered, "cpu")
                else mc_filtered
            )

            ax1.scatter(
                mc_filtered_np[:, 0],
                mc_filtered_np[:, 1],
                s=0.1,
                alpha=0.2,
                label="Monte Carlo Samples",
                color="k",
                zorder=MC_PARTICLES_ZORDER,
            )

            # Plot filtered MC mean
            mc_filtered_mean_x = mc_filtered_np[:, 0].mean()
            mc_filtered_mean_y = mc_filtered_np[:, 1].mean()
            ax1.scatter(
                mc_filtered_mean_x,
                mc_filtered_mean_y,
                s=25,
                alpha=0.9,
                color="red",
                zorder=160,
                marker="x",
                label="Monte Carlo Mean",
            )

        # For contour plot (ax2), use all MC particles
        ax2.scatter(
            mc_np[:, 0],
            mc_np[:, 1],
            s=0.1,
            alpha=0.2,
            label="Monte Carlo Samples",
            color="k",
            zorder=MC_PARTICLES_ZORDER,
        )

        # Plot overall MC mean
        mc_mean_x = mc_np[:, 0].mean()
        mc_mean_y = mc_np[:, 1].mean()
        ax2.scatter(
            mc_mean_x,
            mc_mean_y,
            s=50,
            alpha=0.9,
            color="red",
            zorder=160,
            marker="x",
            label="Monte Carlo Mean",
        )

    # Plot method prediction mean (should be the same for all methods)
    print("Method predictions (tilde_q1):")
    for method_name, method_data in method_map.items():
        if "tilde_q1" in method_data:
            tilde_q1_np = (
                method_data["tilde_q1"].cpu().numpy()
                if hasattr(method_data["tilde_q1"], "cpu")
                else np.array(method_data["tilde_q1"])
            )
            print(f"  {method_name}: {tilde_q1_np.flatten()}")

    common_tilde_point, baseline7_point = select_common_and_mle_tilde_points(method_map)

    if baseline7_point is not None:
        baseline7_mean_x, baseline7_mean_y = baseline7_point
        baseline7_color = METHOD_COLORS["BASELINE7"]

        for ax in (ax1, ax2):
            ax.scatter(
                baseline7_mean_x,
                baseline7_mean_y,
                s=BASELINE7_TILDE_POINT_SIZE,
                alpha=1.0,
                facecolor=baseline7_color,
                # edgecolor="black",
                zorder=200,  # Increased zorder
                marker="*",
                # linewidths=2.0,  # Thicker edge
                label="_nolegend_",
            )

    pred_x, pred_y = common_tilde_point

    # Plot on both subplots
    for ax in [ax1, ax2]:
        ax.scatter(
            pred_x,
            pred_y,
            s=40,
            alpha=1.0,
            color="blue",
            zorder=170,
            marker="+",
            linewidth=5,
            label=r"$\tilde g_1$",
        )

    # Apply shared axis limits using helper function - pass both axes to ensure they get the same limits
    apply_buffered_axis_limits([ax1, ax2], all_points)

    # Create shared legend on top with custom marker sizes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine and deduplicate legend entries with custom sizing
    combined_handles = []
    combined_labels = []
    seen_labels = set()

    for handle, label in zip(handles1 + handles2, labels1 + labels2):
        if label not in seen_labels:
            # Customize marker sizes for specific labels
            if label == "Monte Carlo Mean":
                # Create smaller marker for MC Mean in legend
                import matplotlib.lines as mlines

                custom_handle = mlines.Line2D(
                    [],
                    [],
                    marker="x",
                    color="red",
                    markersize=2,
                    linestyle="None",
                    alpha=0.9,
                )
                combined_handles.append(custom_handle)
            elif label == "Monte Carlo Samples":
                # Create larger marker for Monte Carlo Samples in legend
                import matplotlib.lines as mlines

                custom_handle = mlines.Line2D(
                    [],
                    [],
                    marker="o",
                    color="k",
                    markersize=1,
                    linestyle="None",
                    alpha=0.6,
                )
                combined_handles.append(custom_handle)
            elif label == r"$\tilde g_1$":
                # Create custom marker for Method Prediction in legend
                import matplotlib.lines as mlines

                custom_handle = mlines.Line2D(
                    [],
                    [],
                    marker="+",
                    color="blue",
                    markersize=2,
                    linestyle="None",
                    alpha=1.0,
                    markeredgewidth=2,
                )
                combined_handles.append(custom_handle)
            else:
                # Use original handle with 3x scaling for method markers
                combined_handles.append(handle)

            combined_labels.append(label)
            seen_labels.add(label)

    fig.legend(
        combined_handles,
        combined_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=5,
        fontsize=10,
        markerscale=5,  # Make method markers 3x larger
    )

    # Adjust layout to make more room for legend above plots and bring plots closer
    plt.subplots_adjust(top=0.75, wspace=0.0)

    # Save the comparison plot
    save_path = Path(
        f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}/plots_2d"
    )
    save_path.mkdir(exist_ok=True, parents=True)

    # Get run info for filename
    any_d = next(iter(method_map.values()))
    run_dq0 = any_d.get("dq0", torch.tensor([float("nan")] * 3))
    run_u = any_d.get("u", torch.tensor([float("nan")] * 2))

    dq0_str = format_tensor_for_filename(run_dq0)
    print(dq0_str)
    print(run_dq0)
    u_str = format_tensor_for_filename(run_u)
    print(u_str)
    print(run_u)
    filename = (
        f"fig_2D_comparison_angle_contour_{validation_id}_dq0_{dq0_str}_u_{u_str}"
    )

    plt.tight_layout()
    figure_base_path = save_path / filename
    save_figure_all_formats(
        fig, figure_base_path, 300, "tight", formats=("png", "pdf", "svg")
    )
    plt.close(fig)


def load_validation_data_for_id(
    validation_id, robot_name, confidence_level, boundary_points
):
    """Load all method data for a specific validation ID with polygon data merged.

    Args:
        validation_id: The validation run ID (e.g., "val_isaac_0007")
        robot_name: Robot name (e.g., "Isaac_Jetbot")
        confidence_level: Confidence level (e.g., "over_confident")
        boundary_points: Number of boundary points (e.g., 5000)

    Returns:
        tuple: (method_map, mc_particles, summary_file_path)
            - method_map: dict mapping method names to their data
            - mc_particles: Monte Carlo particles tensor or None
            - summary_file_path: Path to summary statistics file
    """
    from pathlib import Path

    # Load polygon data from summary statistics if available
    summary_file_path = Path(
        f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}/summary_statistics.pt"
    )
    polygon_data = {}
    if summary_file_path.exists():
        try:
            summary_stats = torch.load(
                summary_file_path, map_location="cpu", weights_only=False
            )
            polygon_data = summary_stats.get("polygon_data", {})
            print(f"Loaded polygon data for {len(polygon_data)} methods from summary")
        except Exception as e:
            print(f"Warning: Failed to load polygon data from summary: {e}")

    # Load validation data for the specific ID
    experiment_dir = Path(
        f"data/{robot_name}/experiments/{confidence_level}/boundary_{boundary_points}"
    )
    if not experiment_dir.exists():
        print(f"No experiment directory found: {experiment_dir}")
        return {}, None, summary_file_path

    # Load data for this validation ID
    method_map = {}
    for method_dir in experiment_dir.iterdir():
        if not method_dir.is_dir():
            continue

        method_name = method_dir.name
        validation_metrics_dir = method_dir / "validation" / "metrics"

        if not validation_metrics_dir.exists():
            continue

        # Look for this specific validation ID
        metrics_file = validation_metrics_dir / f"{validation_id}_{method_name}.pt"
        if metrics_file.exists():
            try:
                data = torch.load(metrics_file, map_location="cpu", weights_only=False)

                # Merge polygon data from summary if available
                if (
                    method_name in polygon_data
                    and validation_id in polygon_data[method_name]
                ):
                    polygon_info = polygon_data[method_name][validation_id]
                    data["polygon_coords"] = polygon_info.get("coords")
                    data["area_2d"] = polygon_info.get("area", 0.0)
                    data["iou_vs_mc"] = polygon_info.get("iou_vs_mc", 0.0)
                    print(
                        f"Merged polygon data for {method_name}: area={data['area_2d']:.6f}"
                    )

                method_map[method_name] = data
            except Exception as e:
                print(f"Warning: failed to load {metrics_file}: {e}")

    # Load MC particles from raw validation data
    raw_validation_file = Path(
        f"data/{robot_name}/raw_data/validation/{validation_id}.pt"
    )
    mc_particles = None
    if raw_validation_file.exists():
        try:
            raw_data = torch.load(
                raw_validation_file, map_location="cpu", weights_only=False
            )
            mc_particles = raw_data.get("q1")
        except Exception as e:
            print(f"Warning: Failed to load raw validation data: {e}")

    return method_map, mc_particles, summary_file_path


def setup_2d_plot_axis(ax, title="", xlabel="X (m)", ylabel="Y (m)"):
    """Set up basic properties for a 2D plot axis.

    Args:
        ax: matplotlib axis object
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")


def apply_buffered_axis_limits(ax, all_points, padding_factor=0.05, shared_limits=None):
    """Apply axis limits based on collected points with padding.

    Args:
        ax: matplotlib axis object (or list of axes for shared limits)
        all_points: list of numpy arrays with 2D points
        padding_factor: fraction of range to add as padding
        shared_limits: optional pre-calculated (xlim, ylim) tuple to apply

    Returns:
        tuple: ((x_min, x_max), (y_min, y_max)) calculated limits
    """
    import numpy as np

    # If shared_limits provided, just apply them
    if shared_limits is not None:
        xlim, ylim = shared_limits
        if isinstance(ax, list):
            for a in ax:
                a.set_xlim(xlim)
                a.set_ylim(ylim)
        else:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        return shared_limits

    # Calculate limits from points
    if not all_points:
        return None

    # Combine all points
    combined_points = np.vstack(all_points)
    x_min, x_max = combined_points[:, 0].min(), combined_points[:, 0].max()
    y_min, y_max = combined_points[:, 1].min(), combined_points[:, 1].max()

    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = padding_factor * max(x_range, y_range)

    xlim = (x_min - padding, x_max + padding)
    ylim = (y_min - padding, y_max + padding)

    # Apply to axes
    if isinstance(ax, list):
        for a in ax:
            a.set_xlim(xlim)
            a.set_ylim(ylim)
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    return (xlim, ylim)


def plot_method_contours(
    ax, method_map, buffered_methods=None, collect_points_for_limits=False
):
    """Plot contours/polygons for all methods with consistent styling.

    Args:
        ax: matplotlib axis object
        method_map: dictionary mapping method names to their data
        buffered_methods: list of method names to include in axis limit calculation
        collect_points_for_limits: if True, return points for axis limit calculation

    Returns:
        list: points for axis limits if collect_points_for_limits=True, else empty list
    """
    import numpy as np

    all_points = []
    if buffered_methods is None:
        buffered_methods = ["CLAPS", "BASELINE1", "BASELINE4", "BASELINE5"]

    for method, d in method_map.items():
        color = METHOD_COLORS.get(method, "tab:gray")
        display_name = METHOD_DISPLAY_NAMES.get(method, method)

        # Make CLAPS label bold
        if method == "CLAPS":
            display_name = r"$\mathbf{" + display_name + r"}$"

        # Use pre-computed polygon coordinates
        polygon_coords = d.get("polygon_coords")
        if polygon_coords is not None:
            try:
                # Convert to numpy if needed
                if hasattr(polygon_coords, "cpu"):
                    coords = polygon_coords.cpu().numpy()
                elif hasattr(polygon_coords, "numpy"):
                    coords = polygon_coords.numpy()
                else:
                    coords = np.array(polygon_coords)

                # Ensure coordinates are closed (first point = last point)
                if not np.allclose(coords[0], coords[-1]):
                    coords = np.vstack([coords, coords[0]])

                # Set line styles
                linestyle = "-"
                if "BASELINE3" in method:
                    linestyle = "--"
                elif "BASELINE2" in method:
                    linestyle = ":"
                elif "BASELINE7" in method:
                    linestyle = "-."
                
                # Plot polygon boundary
                ax.plot(
                    coords[:, 0],
                    coords[:, 1],
                    color=color,
                    alpha=0.9,
                    linewidth=3.5,
                    linestyle=linestyle,
                    label=display_name,
                )
                # Add filled area
                ax.fill(
                    coords[:, 0],
                    coords[:, 1],
                    color=color,
                    alpha=0.1,
                )

                # Collect points for axis limits
                if collect_points_for_limits and method in buffered_methods:
                    all_points.append(coords)

            except Exception as e:
                print(f"Warning: Failed to use polygon_coords for {method}: {e}")
                # Fallback to scatter plot if available
                boundary = d.get("boundary_SS_pts")
                if boundary is not None and boundary.ndim >= 2:
                    points_2d = boundary[:, :2]
                    ax.scatter(
                        points_2d[:, 0],
                        points_2d[:, 1],
                        s=2,
                        alpha=0.5,
                        color=color,
                        label=display_name,
                    )
                    if collect_points_for_limits and method in buffered_methods:
                        all_points.append(points_2d)
        else:
            # Fallback to scatter plot when no polygon coordinates available
            boundary = d.get("boundary_SS_pts")
            if boundary is not None and boundary.ndim >= 2:
                points_2d = boundary[:, :2]
                ax.scatter(
                    points_2d[:, 0],
                    points_2d[:, 1],
                    s=2,
                    alpha=0.5,
                    color=color,
                    label=display_name,
                )
                if collect_points_for_limits and method in buffered_methods:
                    all_points.append(points_2d)

    return all_points if collect_points_for_limits else []


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

    import argparse

    parser = argparse.ArgumentParser(
        description="Extract and analyze CLAPS validation metrics"
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        required=True,
        help="Robot system name (e.g., Real_MBot, Isaac_Jetbot)",
    )
    parser.add_argument(
        "--boundary-points",
        type=int,
        required=True,
        help="Number of boundary points used in mesh generation",
    )
    parser.add_argument(
        "--confidence-level",
        type=str,
        required=True,
        choices=["over_confident", "under_confident", "default"],
        help="Confidence level",
    )
    parser.add_argument(
        "--recompute",
        default=False,
        action="store_true",
        help="Force recomputation of all metrics (overwrite existing files)",
    )
    parser.add_argument(
        "--stats-only",
        default=False,
        action="store_true",
        help="Only compute and display statistics, no plots or tables",
    )
    parser.add_argument(
        "--skip-2d",
        default=False,
        action="store_true",
        help="Skip 2D projection plots (many files, slower)",
    )
    parser.add_argument(
        "--skip-3d",
        default=True,
        action="store_true",
        help="Skip 3D plots with multiple viewing angles (many files, very slow)",
    )
    parser.add_argument(
        "--mpl-3d",
        default=True,
        action="store_true",
        help="Generate Matplotlib 3D plots (paper-style). Uses azimuth 315 when --validation-id=0301. Enabled by default.",
    )
    parser.add_argument(
        "--skip-mpl-3d",
        default=False,
        action="store_true",
        help="Skip Matplotlib 3D plots",
    )
    parser.add_argument(
        "--skip-debug",
        default=False,
        action="store_true",
        help="Skip debugging plots showing volume ratio vs motion magnitude correlations",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Enable verbose logging output",
    )
    parser.add_argument(
        "--dry-run",
        default=False,
        action="store_true",
        help="Process only first few validation cases for testing",
    )

    # Phase control flags
    parser.add_argument(
        "--mc-only",
        default=False,
        action="store_true",
        help="Phase 1: Only compute MC polygons, save to disk, then exit",
    )
    parser.add_argument(
        "--methods-only",
        default=False,
        action="store_true",
        help="Phase 2: Only process methods (load MC polygons from disk), compute metrics, save results, then exit",
    )
    parser.add_argument(
        "--parallel-methods",
        action="store_true",
        help="Process method metrics in parallel within Phase 2",
    )
    parser.add_argument(
        "--no-parallel-methods",
        dest="parallel_methods",
        action="store_false",
        help="Force sequential method processing (overrides mode default)",
    )
    parser.set_defaults(parallel_methods=None)
    parser.add_argument(
        "--method-workers",
        type=int,
        default=None,
        help="Max workers for --parallel-methods (default: min(cpu_count, MAX_WORKERS))",
    )
    parser.add_argument(
        "--plots-only",
        default=False,
        action="store_true",
        help="Phase 3: Only generate plots (load saved results from disk)",
    )
    parser.add_argument(
        "--table-only",
        default=False,
        action="store_true",
        help=(
            "Generate the LaTeX/PDF results table without plots; combine with --methods-only to run aggregation then exit"
        ),
    )
    parser.add_argument(
        "--validation-id",
        type=str,
        default=None,
        help="Process only a specific validation ID (e.g., '0300')",
    )
    parser.add_argument(
        "--skip-coverage-check",
        default=False,
        action="store_true",
        help="Skip coverage criteria check (process any validation ID)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["paper-plots", "all-plots"],
        default=None,
        help="Execution mode: 'paper-plots' (fast, specific plots) or 'all-plots' (comprehensive)",
    )

    args = parser.parse_args()

    # Validate phase flags - only one of the main phases may be active at a time
    # BUT if mode is specified, we ignore these checks or handle them differently
    phase_flags = [args.mc_only, args.methods_only, args.plots_only]
    if args.mode is None and sum(phase_flags) > 1:
        print(
            "❌ Error: Only one phase flag can be used at a time (--mc-only, --methods-only, or --plots-only)"
        )
        sys.exit(1)

    logger = setup_logging(verbose=args.verbose)
    logger.info(
        f"🚀 Running metrics pipeline for {args.robot_type} boundary_{args.boundary_points} ({args.confidence_level})..."
    )
    if args.mode:
        logger.info(f"    Mode: {args.mode}")
    else:
        logger.info(
            f"    Options: plot_2d={not args.skip_2d}, plot_3d={not args.skip_3d}, mpl_3d={args.mpl_3d}, debug_plots={not args.skip_debug}"
        )

    if args.mode:
        # Consolidated pipeline based on file existence
        reports_dir = Path(
            f"data/{args.robot_type}/reports/{args.confidence_level}/boundary_{args.boundary_points}"
        )
        mc_polygons_file = reports_dir / "mc_polygons.pt"
        summary_file = reports_dir / "summary_statistics.pt"

        force_recompute = args.recompute

        # Detect validation count drift to avoid stale summaries when splits change
        validation_mismatch = False
        if summary_file.exists():
            validation_mismatch, current_val_count, recorded_val_count = (
                validation_count_mismatch(summary_file, args.robot_type)
            )
            if validation_mismatch:
                logger.info(
                    f"🔄 Validation count changed (summary={recorded_val_count}, current={current_val_count}); "
                    "recomputing MC polygons and method summaries."
                )

        # Phase 1: MC Polygons
        if force_recompute or validation_mismatch or not mc_polygons_file.exists():
            phase1_reason = "force-recompute" if force_recompute else (
                "count mismatch" if validation_mismatch else "missing file"
            )
            logger.info(f"🔶 Phase 1: Computing MC polygons ({phase1_reason})...")
            success = compute_mc_polygons_phase(
                args.robot_type,
                args.confidence_level,
                args.boundary_points,
                logger,
                args.dry_run,
            )
            if not success:
                logger.error("❌ Phase 1 failed")
                sys.exit(1)
        else:
            logger.info("⏩ Phase 1: MC polygons already exist (skipping)")

        # Phase 2: Methods
        if force_recompute or validation_mismatch or not summary_file.exists():
            phase2_reason = "force-recompute" if force_recompute else (
                "count mismatch" if validation_mismatch else "missing summary"
            )
            logger.info(f"🚀 Phase 2: Processing methods ({phase2_reason})...")
            # For paper-plots, we might want parallel methods to be faster
            parallel = resolve_method_parallelism(
                mode=args.mode, parallel_methods_flag=args.parallel_methods
            )
            
            success = process_methods_phase(
                args.robot_type,
                args.confidence_level,
                args.boundary_points,
                logger,
                True,  # recompute metrics whenever we rerun Phase 2
                args.dry_run,
                parallel,
                args.method_workers,
            )
            if not success:
                logger.error("❌ Phase 2 failed")
                sys.exit(1)
        else:
            logger.info("⏩ Phase 2: Method summary already exists (skipping)")

        # Phase 3: Plots
        logger.info(f"🎨 Phase 3: Generating plots (mode={args.mode})...")
        success = generate_plots_phase(
            args.robot_type,
            args.confidence_level,
            args.boundary_points,
            not args.skip_2d,
            not args.skip_3d,
            args.mpl_3d and not args.skip_mpl_3d,
            not args.skip_debug, # Passed but filtered inside based on mode
            logger,
            mode=args.mode,
        )
        if success:
            logger.info(f"✅ Pipeline completed! ({args.mode})")
            sys.exit(0)
        else:
            logger.error("❌ Phase 3 failed")
            sys.exit(1)

    elif args.mc_only:
        logger.info("🔶 Phase 1: Computing MC polygons only...")
        success = compute_mc_polygons_phase(
            args.robot_type,
            args.confidence_level,
            args.boundary_points,
            logger,
            args.dry_run,
        )
        if success:
            logger.info("✅ Phase 1 completed! MC polygons saved to disk.")
            sys.exit(0)
        else:
            logger.error("❌ Phase 1 failed")
            sys.exit(1)
    elif args.methods_only:
        logger.info("🚀 Phase 2: Processing methods only...")
        parallel_methods = resolve_method_parallelism(
            mode=args.mode, parallel_methods_flag=args.parallel_methods
        )
        success = process_methods_phase(
            args.robot_type,
            args.confidence_level,
            args.boundary_points,
            logger,
            args.recompute,
            args.dry_run,
            parallel_methods,
            args.method_workers,
        )
        if success:
            logger.info(
                "✅ Phase 2 completed! Method results and metrics saved to disk."
            )
            if args.table_only:
                summary_path = Path(
                    f"data/{args.robot_type}/reports/{args.confidence_level}/boundary_{args.boundary_points}/summary_statistics.pt"
                )
                if summary_path.exists():
                    logger.info("🧾 Table-only mode: generating LaTeX/PDF table...")
                    generate_latex_table_from_summary(str(summary_path), logger)
                else:
                    logger.error(
                        f"❌ Summary file not found for table-only mode: {summary_path}"
                    )
            sys.exit(0)
        else:
            logger.error("❌ Phase 2 failed")
            sys.exit(1)
    elif args.plots_only:
        logger.info("🎨 Phase 3: Generating plots only...")
        # Respect skip-debug flag in plots-only mode
        debug_plots = not args.skip_debug
        if debug_plots:
            logger.info("    Debug plots enabled")
        else:
            logger.info("    Debug plots disabled (--skip-debug)")
        success = generate_plots_phase(
            args.robot_type,
            args.confidence_level,
            args.boundary_points,
            not args.skip_2d,
            not args.skip_3d,
            args.mpl_3d and not args.skip_mpl_3d,
            debug_plots,
            logger,
        )
        if success:
            logger.info("✅ Phase 3 completed! All plots generated.")
            sys.exit(0)
        else:
            logger.error("❌ Phase 3 failed")
            sys.exit(1)
    elif args.table_only:
        logger.info("🧾 Table-only mode: using existing summary_statistics.pt")
        summary_path = Path(
            f"data/{args.robot_type}/reports/{args.confidence_level}/boundary_{args.boundary_points}/summary_statistics.pt"
        )
        if not summary_path.exists():
            logger.error(
                f"❌ Summary file not found: {summary_path}. Run --methods-only first."
            )
            sys.exit(1)

        try:
            generate_latex_table_from_summary(str(summary_path), logger)
            logger.info("✅ Table generated successfully")
            sys.exit(0)
        except Exception as exc:
            logger.error(f"❌ Table-only mode failed: {exc}")
            sys.exit(1)
    else:
        # No phase specified - show error and exit
        logger.error("❌ Error: Must specify one of the phase flags:")
        logger.error("   --mc-only     (Phase 1: Compute MC polygons)")
        logger.error("   --methods-only (Phase 2: Process methods)")
        logger.error("   --plots-only   (Phase 3: Generate plots)")
        sys.exit(1)
