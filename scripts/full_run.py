import warnings

import numpy as np
import torch

# Suppress torch deprecation warning
warnings.filterwarnings("ignore", "torch.set_default_tensor_type.*", UserWarning)

from luis_utils.load import script_starter

script_starter(precision="double", device="cuda")

import multiprocessing as mp
import time
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import pyvista as pv
import yaml
from helpers import (
    _l2_norm,
    estimate_invariant_error_covariance,
    estimate_mean_and_covariance,
    run_calibration,
    run_validation,
)
from luis_utils.algorithms import (
    ROBOT_QSPACE_TO_FACTORY,
    ROBOT_TO_KEY,
    Algorithm,
    ConfigMapper,
    NonconformityScore,
    Q_Space,
    RobotType,
    VelocityMapper,
)
from luis_utils.env import SecondOrderEnv
from luis_utils.estimators.ekf import EKFEstimator
from luis_utils.gaussians import (
    ball_volume,
    confidence_hyperellipsoid_volume,
    critical_mahalanobis_distance,
    fibonacci_3Dsphere_grid,
    sample_confidence_ellipsoid_boundary,
)
from luis_utils.vecops import wrap_angle
from pymatlie.se2 import SE2
from scipy.spatial import ConvexHull
from tqdm import tqdm

# METHODS
methods = {
    "CLAPS": {
        "space": Q_Space.LIE,
        "calibrate": True,
        "r_metric": NonconformityScore.MAHALANOBIS,
        "name": "CLAPS",
    },
    "BASELINE1": {
        "space": Q_Space.STATE_SPACE,
        "calibrate": True,
        "r_metric": NonconformityScore.MAHALANOBIS,
        "name": "SS EKF + CP",
    },
    "BASELINE2": {
        "space": Q_Space.LIE,
        "calibrate": True,
        "r_metric": NonconformityScore.L2,
        "name": "Lie PP + CP",
    },
    "BASELINE3": {
        "space": Q_Space.STATE_SPACE,
        "calibrate": True,
        "r_metric": NonconformityScore.L2,
        "name": "SS PP + CP",
    },
    "BASELINE4": {
        "space": Q_Space.LIE,
        "calibrate": False,
        "r_metric": NonconformityScore.MAHALANOBIS,
        "name": "InEKF",
    },
    "BASELINE5": {
        "space": Q_Space.STATE_SPACE,
        "calibrate": False,
        "r_metric": NonconformityScore.MAHALANOBIS,
        "name": "SS EKF",
    },
    "BASELINE6": {
        "space": Q_Space.LIE,
        "calibrate": True,
        "r_metric": NonconformityScore.MAHALANOBIS,
        "name": "InEKF + 2M",
        "calibration_strategy": "covariance_fit",
    },
    "BASELINE7": {
        "space": Q_Space.LIE,
        "calibrate": True,
        "r_metric": NonconformityScore.MAHALANOBIS,
        "name": "InEKF + MLE",
        "calibration_strategy": "mean_covariance_fit",
    },
}

delaunay_methods = ["CLAPS", "BASELINE4", "BASELINE6", "BASELINE7"]


def get_experiment_dir(
    robot_name: str, confidence_level: str, boundary_points: int
) -> Path:
    """Get the experiment directory path for given parameters."""
    return Path(
        f"data/{robot_name}/experiments/{confidence_level}/boundary_{boundary_points}"
    )


def get_reports_dir(
    robot_name: str, confidence_level: str, boundary_points: int
) -> Path:
    """Get the reports directory path for given parameters."""
    return Path(
        f"data/{robot_name}/reports/{confidence_level}/boundary_{boundary_points}"
    )


def ensure_dir_structure(base_path: Path, method: str = None) -> dict:
    """Create directory structure and return important paths."""
    paths = {
        "base": base_path,
        "calibration": base_path / "calibration",
        "validation": base_path / "validation" / "metrics",
        "visualizations": base_path / "validation" / "visualizations",
    }

    if method:
        method_base = base_path / method
        paths.update(
            {
                "base": method_base,
                "calibration": method_base / "calibration",
                "validation": method_base / "validation" / "metrics",
                "visualizations": method_base / "validation" / "visualizations",
            }
        )

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def clean_validation_data(experiment_dir: Path, method: str) -> None:
    """Clean validation data for a specific method in experiments folder."""
    method_dir = experiment_dir / method
    validation_dir = method_dir / "validation"

    if not validation_dir.exists():
        return

    cleaned_count = 0

    metrics_dir = validation_dir / "metrics"
    if metrics_dir.exists():
        for f in metrics_dir.glob("*.pt"):
            f.unlink()
            cleaned_count += 1

    viz_dir = validation_dir / "visualizations"
    if viz_dir.exists():
        import shutil

        for f in viz_dir.glob("*.rrd"):
            if f.is_file():
                f.unlink()
                cleaned_count += 1

    if cleaned_count > 0:
        print(f"Cleaned {cleaned_count} validation files for {method}")


def generate_boundary_SS_pts(
    *,
    method_cfg: dict,
    params: dict,
    boundary_points: int,
    conf1_hat: torch.Tensor,
    vel1_hat: torch.Tensor | None,
    xi_sigma_calibrated: torch.Tensor | None,
    loaded_scaling_factor: float | torch.Tensor | None,
    method_system_key,
    gt_system_key,
) -> np.ndarray:
    """Generate boundary points in state space for mesh building.

    - If space is Lie: sample boundary in algebra and map via exp to group, then to state space.
    - If space is State Space: sample boundary directly in state space around mean.
    """
    n_points = boundary_points
    device = conf1_hat.device

    if method_cfg["space"] == Q_Space.LIE:
        if method_cfg["r_metric"] == NonconformityScore.MAHALANOBIS:
            assert (
                xi_sigma_calibrated is not None
            ), "xi_sigma_calibrated required for Mahalanobis boundary"
            boundary_xi_pts = sample_confidence_ellipsoid_boundary(
                n_points=n_points,
                mean=(
                    torch.zeros_like(vel1_hat[0, :])
                    if vel1_hat is not None
                    else torch.zeros(3, device=device)
                ),
                covariance=xi_sigma_calibrated,
                confidence_level=1 - params["failure_rate"],
            )
        elif method_cfg["r_metric"] == NonconformityScore.L2:
            radius = (
                float(loaded_scaling_factor)
                if torch.is_tensor(loaded_scaling_factor)
                else loaded_scaling_factor
            )
            unit_pts = fibonacci_3Dsphere_grid(n_points=n_points)
            boundary_xi_pts = unit_pts * radius
        else:
            raise ValueError(f"Invalid r_metric: {method_cfg['r_metric']}")

        # Keep yaw within principal branch for diffeomorphism
        boundary_xi_pts[:, 2] = wrap_angle(boundary_xi_pts[:, 2])

        # Map to Lie group, then to state space
        if params["error_form"] == "RI":
            boundary_G_pts = SE2.exp(boundary_xi_pts) @ conf1_hat[0]
        elif params["error_form"] == "LI":
            boundary_G_pts = conf1_hat[0] @ SE2.exp(boundary_xi_pts)
        else:
            raise ValueError(f"Invalid error_form: {params['error_form']}")

        boundary_SS_pts = (
            ConfigMapper.map(
                q=boundary_G_pts, from_key=method_system_key, to_key=gt_system_key
            )
            .cpu()
            .numpy()
        )
        return boundary_SS_pts

    elif method_cfg["space"] == Q_Space.STATE_SPACE:
        # Center in state space
        center_SS = ConfigMapper.map(
            q=conf1_hat[0:1, ...], from_key=method_system_key, to_key=gt_system_key
        )[0]

        if method_cfg["r_metric"] == NonconformityScore.MAHALANOBIS:
            assert (
                xi_sigma_calibrated is not None
            ), "xi_sigma_calibrated required for Mahalanobis boundary"
            boundary_diff = sample_confidence_ellipsoid_boundary(
                n_points=n_points,
                mean=torch.zeros(3, device=device),
                covariance=xi_sigma_calibrated,
                confidence_level=1 - params["failure_rate"],
            )
        elif method_cfg["r_metric"] == NonconformityScore.L2:
            radius = (
                float(loaded_scaling_factor)
                if torch.is_tensor(loaded_scaling_factor)
                else loaded_scaling_factor
            )
            unit_pts = fibonacci_3Dsphere_grid(n_points=n_points)
            boundary_diff = unit_pts * radius
        else:
            raise ValueError(f"Invalid r_metric: {method_cfg['r_metric']}")

        boundary_SS_pts = (center_SS + boundary_diff).cpu().numpy()
        return boundary_SS_pts

    else:
        raise ValueError(f"Invalid space: {method_cfg['space']}")


@contextmanager
def record_time(metrics: dict, key: str, sync_cuda: bool = True):
    """Record execution time by appending to a list."""
    try:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    start_time = time.perf_counter()
    try:
        yield
    finally:
        try:
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        elapsed = time.perf_counter() - start_time
        if key not in metrics:
            metrics[key] = []
        metrics[key].append(elapsed)


def process_single_validation_file(validation_file_path, shared_params):
    """Process a single validation file and return metrics.

    Args:
        validation_file_path: Path to the validation .pt file
        shared_params: Dictionary containing all shared parameters

    Returns:
        dict: Metrics dictionary for this validation file
    """
    from pathlib import Path

    import numpy as np
    import pyvista as pv
    import torch
    from helpers import _l2_norm
    from luis_utils.algorithms import (
        ROBOT_QSPACE_TO_FACTORY,
        ConfigMapper,
        NonconformityScore,
        Q_Space,
        VelocityMapper,
    )
    from luis_utils.gaussians import (
        ball_volume,
        confidence_hyperellipsoid_volume,
        mahalanobis_distance,
        points_in_confidence_hyperellipsoid,
    )
    from luis_utils.plotting import visualize_confidence_rerun
    from pymatlie.se2 import SE2
    from scipy.spatial import ConvexHull

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    p = Path(validation_file_path)
    method_cfg = shared_params["method_cfg"]
    method_paths = shared_params["method_paths"]
    loaded_scaling_factor = shared_params["loaded_scaling_factor"]
    loaded_covariance = shared_params.get("loaded_covariance")
    loaded_bias = shared_params["loaded_bias"]
    params = shared_params["params"]
    robot_name = shared_params["robot_name"]
    boundary_points = shared_params["boundary_points"]
    gt_system_key = shared_params["gt_system_key"]
    method_system_key = shared_params["method_system_key"]

    inertia_matrix = shared_params["inertia_matrix"]
    if torch.is_tensor(inertia_matrix):
        inertia_matrix = inertia_matrix.to(device)

    Q_cont = shared_params["Q_cont"]
    if torch.is_tensor(Q_cont):
        Q_cont = Q_cont.to(device)

    # Recreate system objects in each process (for multiprocessing compatibility)
    from luis_utils.env import SecondOrderEnv
    from luis_utils.estimators.ekf import EKFEstimator

    propagator_config = shared_params["approximate_propagator_config"]

    inertia_matrix_estimate = propagator_config["inertia_matrix_estimate"]
    if torch.is_tensor(inertia_matrix_estimate):
        inertia_matrix_estimate = inertia_matrix_estimate.to(device)

    # Recreate method system
    method_system = ROBOT_QSPACE_TO_FACTORY[method_system_key](
        inertia_matrix=inertia_matrix_estimate
    )

    method_env = SecondOrderEnv(
        system=method_system,
        physics_dt=propagator_config["physics_dt"],
        max_steps_per_episode=1,
        decimation=propagator_config["decimation"],
        integrator_name=propagator_config["planner_integrator"],
    )

    # Create fresh EKF for this worker
    approximate_propagator = EKFEstimator(
        system=method_system,
        dt=propagator_config["physics_dt"],
        integrator=method_env.integrator,
        error_form=propagator_config["error_form"],
    )
    method_name = shared_params["method_name"]
    delaunay_methods = shared_params["delaunay_methods"]
    calibration_strategy = shared_params.get("calibration_strategy", "scaling")

    val_time_metrics = {}
    data = torch.load(p, map_location=torch.get_default_device(), weights_only=False)
    # Convert Real_MBot accelerations to forces for consistent storage
    u_metrics = data["u"][0].clone()
    if params["simulator"] == "real_robot":
        # Real_MBot data contains accelerations - convert to forces
        u_metrics[0] *= inertia_matrix[0, 0]  # linear acceleration to force
        u_metrics[1] *= inertia_matrix[2, 2]  # angular acceleration to torque

    metrics = {
        "q0": data["q0"][0],
        "q1": data["q1"][0],
        "dq0": data["dq0"][0],
        "dq1": data["dq1"][0],
        "u": u_metrics,
    }

    print(f"Evaluating {p}")

    dcal_q0_modelForm = ConfigMapper.map(
        from_key=gt_system_key, to_key=method_system_key, q=data["q0"]
    )
    dcal_q1_modelForm = ConfigMapper.map(
        from_key=gt_system_key, to_key=method_system_key, q=data["q1"]
    )
    dcal_dq0_modelForm = VelocityMapper.map(
        from_key=gt_system_key,
        to_key=method_system_key,
        q=data["q0"],
        dq_or_v=data["dq0"],
    )
    dcal_dq1_modelForm = VelocityMapper.map(
        from_key=gt_system_key,
        to_key=method_system_key,
        q=data["q1"],
        dq_or_v=data["dq1"],
    )

    q1 = data["q1"].clone()

    with record_time(val_time_metrics, "dynamics_predict_total"):
        # Convert Real_MBot accelerations to forces for consistent dynamics
        u_data_val = data["u"][0:1].clone()
        if params["simulator"] == "real_robot":
            # Real_MBot data contains accelerations - convert to forces
            u_data_val[:, 0] *= inertia_matrix[0, 0]  # linear acceleration to force
            u_data_val[:, 1] *= inertia_matrix[2, 2]  # angular acceleration to torque

        s_estimate = {
            "P": torch.zeros(1, 6, 6),
            "pose": dcal_q0_modelForm[0:1],
            "velocity": dcal_dq0_modelForm[0:1],
            "Q": Q_cont,
        }
        s_estimate = approximate_propagator.predict_planning_step(
            s_estimate, u_data_val, params["decimation"]
        )
        xi_sigma_approx = s_estimate["P"][:, :3, :3]

        assert xi_sigma_approx.shape == (
            1,
            3,
            3,
        ), f"xi_sigma_approx.shape: {xi_sigma_approx.shape}"

    conf1_hat = s_estimate["pose"]
    vel1_hat = s_estimate["velocity"]
    conf1_hat_pre_bias = conf1_hat.clone()

    if loaded_bias is not None:
        bias_for_correction = loaded_bias.to(conf1_hat.device)
        if bias_for_correction.ndim == 1:
            bias_for_correction = bias_for_correction.unsqueeze(0)
        # Apply learned mean error in the direction predicted->true
        conf1_hat = SE2.right_plus(conf1_hat, bias_for_correction)
        s_estimate["pose"] = conf1_hat  # keep downstream boundary generation bias-corrected

    if method_cfg["space"] == Q_Space.STATE_SPACE:
        metrics["mean_pred"] = conf1_hat[0]
    elif method_cfg["space"] == Q_Space.LIE:
        metrics["mean_pred"] = SE2.map_configuration_to_q(conf1_hat)[0]

    if method_cfg["space"] == Q_Space.STATE_SPACE:
        metrics["tilde_q1_pre_bias"] = conf1_hat_pre_bias[0].cpu().numpy()
    elif method_cfg["space"] == Q_Space.LIE:
        metrics["tilde_q1_pre_bias"] = (
            ConfigMapper.map(
                from_key=method_system_key,
                to_key=gt_system_key,
                q=conf1_hat_pre_bias[0:1, ...],
            )
            .cpu()
            .numpy()
        )

    if method_cfg["space"] == Q_Space.STATE_SPACE:
        min_eig = torch.linalg.eigvals(xi_sigma_approx[0]).real.min()
        if min_eig < 1e-5:  # Regularization to avoid numerical instability
            xi_sigma_approx = (
                xi_sigma_approx + torch.eye(3, device=xi_sigma_approx.device) * 1e-5
            )

    with record_time(val_time_metrics, "error_total"):
        if method_cfg["space"] == Q_Space.LIE:
            if params["error_form"] == "RI":
                err = SE2.log(
                    SE2.right_invariant_error(
                        true_state=dcal_q1_modelForm, estimated_state=conf1_hat
                    )
                )
            elif params["error_form"] == "LI":
                err = SE2.right_minus(g_start=conf1_hat, g_end=dcal_q1_modelForm)
        elif method_cfg["space"] == Q_Space.STATE_SPACE:
            err = dcal_q1_modelForm - conf1_hat
        else:
            raise ValueError(f"Invalid space: {method_cfg['space']}")

    with record_time(val_time_metrics, "online_calibration_total"):
        xi_sigma_calibrated = None
        if method_cfg["r_metric"] == NonconformityScore.MAHALANOBIS:
            if calibration_strategy in ("covariance_fit", "mean_covariance_fit"):
                if loaded_covariance is None:
                    raise ValueError(
                        "Loaded covariance is required for covariance_fit or mean_covariance_fit strategy"
                    )
                xi_sigma_calibrated = loaded_covariance.to(device)
                if xi_sigma_calibrated.ndim == 2:
                    xi_sigma_calibrated = xi_sigma_calibrated.unsqueeze(0)
            else:
                if loaded_scaling_factor is None:
                    raise ValueError("Scaling factor required for scaling strategy")
                xi_sigma_calibrated = xi_sigma_approx * loaded_scaling_factor

    metrics["tilde_q1"] = (
        ConfigMapper.map(
            from_key=method_system_key,
            to_key=gt_system_key,
            q=conf1_hat[0:1, ...],
        )
        .cpu()
        .numpy()
    )

    if method_cfg["r_metric"] == NonconformityScore.MAHALANOBIS:
        if xi_sigma_calibrated is None:
            raise ValueError("Mahalanobis metric requires calibrated covariance")
        if xi_sigma_calibrated.ndim == 2:
            xi_sigma_calibrated = xi_sigma_calibrated.unsqueeze(0)

        if xi_sigma_calibrated.shape[0] == 1:
            xi_sigma_calibrated = xi_sigma_calibrated[0]
        elif torch.all(xi_sigma_calibrated == xi_sigma_calibrated[0:1]):
            xi_sigma_calibrated = xi_sigma_calibrated[0]
        else:
            raise ValueError("Calibrated covariance must be shared across batch")

        assert xi_sigma_calibrated.shape == (3, 3)

    if method_cfg["r_metric"] == NonconformityScore.MAHALANOBIS:
        mask_lie_algebra, metrics["empirical_coverage_LA"] = (
            points_in_confidence_hyperellipsoid(
                difference=err,
                covariance=xi_sigma_calibrated,
                failure_rate=params["failure_rate"],
            )
        )
        mask_lie_algebra = mask_lie_algebra.cpu().numpy()
    elif method_cfg["r_metric"] == NonconformityScore.L2:
        distances = _l2_norm(err, dim=-1)
        mask_lie_algebra = (distances <= loaded_scaling_factor).cpu().numpy()
        metrics["empirical_coverage_LA"] = float(mask_lie_algebra.mean())

    with record_time(val_time_metrics, "sample_boundary_total"):
        boundary_SS_pts = generate_boundary_SS_pts(
            method_cfg=method_cfg,
            params=params,
            boundary_points=boundary_points,
            conf1_hat=s_estimate["pose"],
            vel1_hat=s_estimate.get("velocity"),
            xi_sigma_calibrated=xi_sigma_calibrated,
            loaded_scaling_factor=loaded_scaling_factor,
            method_system_key=method_system_key,
            gt_system_key=gt_system_key,
        )

    with record_time(val_time_metrics, "mesh_fitting_total"):
        if method_name in delaunay_methods:
            from CGAL.CGAL_Kernel import Point_3
            from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3

            cgal_pts = [
                Point_3(float(x), float(y), float(z)) for x, y, z in boundary_SS_pts
            ]
            print(f"Computing mesh volume for: {method_cfg['name']}")

            with record_time(val_time_metrics, "cgal_delaunay_only"):
                dt = Delaunay_triangulation_3()
                dt.insert(cgal_pts)

            print("🚀 Using CGAL triangulation + PyVista delaunay")

            with record_time(val_time_metrics, "pyvista_processing"):
                # Extract vertices from CGAL triangulation
                vertices_list = []
                for vertex in dt.finite_vertices():
                    point = vertex.point()
                    vertices_list.append(
                        [
                            float(point.x()),
                            float(point.y()),
                            float(point.z()),
                        ]
                    )

                all_vertices_array = np.array(vertices_list)
                unique_vertices = np.unique(all_vertices_array, axis=0)

            with record_time(val_time_metrics, "pyvista_delaunay"):
                cgal_vertices_cloud = pv.wrap(unique_vertices)
                pyvista_from_cgal = cgal_vertices_cloud.delaunay_3d()
                mesh_SS = pyvista_from_cgal.extract_surface()

            metrics["total_volume_mesh"] = mesh_SS.volume

            print(
                f"  {method_cfg['name']}: Volume={mesh_SS.volume:.6f}, Watertight={mesh_SS.n_open_edges == 0}"
            )

    if method_name in delaunay_methods:
        print("Open edges:", mesh_SS.n_open_edges)
        print("Watertight?", mesh_SS.n_open_edges == 0)
        # Ensure normals are computed for Delaunay meshes
        if not hasattr(mesh_SS, "normals") or mesh_SS.normals is None:
            mesh_SS.compute_normals(
                consistent_normals=True, auto_orient_normals=True, inplace=True
            )

        dist_boundaryPts2boundaryMesh_obj = pv.wrap(
            boundary_SS_pts
        ).compute_implicit_distance(mesh_SS)
        dist_boundaryPts2boundaryMesh = np.abs(
            dist_boundaryPts2boundaryMesh_obj["implicit_distance"]
        )

        metrics["max_point_dist2mesh"] = dist_boundaryPts2boundaryMesh.max()
        metrics["mean_point_dist2mesh"] = dist_boundaryPts2boundaryMesh.mean()

        metrics["total_volume_mesh"] = mesh_SS.volume

    else:
        # For analytical methods, compute analytical volume only
        print(f"Computing analytical volume for: {method_cfg['name']}")
        metrics["max_point_dist2mesh"] = 0.0
        metrics["mean_point_dist2mesh"] = 0.0

        if (
            method_cfg["r_metric"] == NonconformityScore.MAHALANOBIS
        ):  # Using mahalanobis distance
            mask_state_space, empirical_coverage_SS = (
                points_in_confidence_hyperellipsoid(
                    difference=err,
                    covariance=xi_sigma_calibrated,
                    failure_rate=params["failure_rate"],
                )
            )
            mask_state_space = mask_state_space.cpu().numpy()
            # Compute Cholesky factor for volume calculation
            cholesky_factor = torch.linalg.cholesky(xi_sigma_calibrated)
            analytical_volume = confidence_hyperellipsoid_volume(
                cholesky_factor=cholesky_factor,
                failure_rate=params["failure_rate"],
            ).item()
            metrics["total_volume_mesh"] = analytical_volume

            # Store analytical volume
            metrics["analytical_volume"] = analytical_volume

            print(f"ANALYTICAL VOLUME for {method_cfg['name']}:")
            print(f"  Analytical Ellipsoid: {analytical_volume:.6f}")

        elif method_cfg["r_metric"] == NonconformityScore.L2:
            distances = _l2_norm(err, dim=-1)
            mask_state_space = (distances <= loaded_scaling_factor).cpu().numpy()
            analytical_volume = ball_volume(loaded_scaling_factor, 3).item()
            metrics["total_volume_mesh"] = analytical_volume

            # Store analytical volume
            metrics["analytical_volume"] = analytical_volume

            print(f"ANALYTICAL VOLUME for {method_cfg['name']}:")
            print(f"  Analytical Ball: {analytical_volume:.6f}")

    # Handle coverage computation based on method
    if method_name in delaunay_methods:
        # For mesh-based approaches, compute coverage from mesh
        q1_numpy = q1.cpu().numpy()  # Convert once outside timer
        with record_time(val_time_metrics, "check_points_in_mesh_total"):
            pt_wrap = pv.wrap(q1_numpy)
            mask_state_space = (
                pt_wrap.compute_implicit_distance(mesh_SS)["implicit_distance"] < 1e-5
            )

    # Coverage is already computed for direct computation methods
    metrics["empirical_coverage_SS"] = mask_state_space.mean().item()

    # For replotting
    metrics["boundary_SS_pts"] = boundary_SS_pts
    metrics["mask_state_space"] = mask_state_space  # Save the inside/outside mask

    if not (method_name in delaunay_methods):
        import pyvista as pv
        from scipy.spatial import ConvexHull

        convex_hull_obj = ConvexHull(boundary_SS_pts)

        viz_convex_hull = pv.PolyData(
            convex_hull_obj.points,
            faces=np.column_stack(
                [
                    np.full(len(convex_hull_obj.simplices), 3),  # 3 vertices per face
                    convex_hull_obj.simplices,
                ]
            ).flatten(),
        )
    else:
        viz_convex_hull = mesh_SS

    # Save mesh data for later use (voxelization, etc.)
    metrics["mesh_vertices"] = viz_convex_hull.points
    metrics["mesh_faces"] = viz_convex_hull.faces
    metrics["mesh_volume"] = viz_convex_hull.volume
    metrics["mesh_n_vertices"] = viz_convex_hull.n_points
    metrics["mesh_n_faces"] = viz_convex_hull.n_cells
    metrics["mesh_is_watertight"] = viz_convex_hull.n_open_edges == 0
    metrics["mesh_n_open_edges"] = viz_convex_hull.n_open_edges

    q1_for_viz = q1_numpy if "q1_numpy" in locals() else q1.cpu().numpy()
    visualization_path = method_paths["visualizations"] / f"{p.stem}_{method_name}"
    visualize_confidence_rerun(
        boundary_mesh=viz_convex_hull,
        sampled_points_MC=q1_for_viz,
        convex_hull=viz_convex_hull,
        mask=mask_state_space,
        boundary_pts=boundary_SS_pts,
        mask_algebra=mask_lie_algebra,
        save_path=str(visualization_path),
        marker_size=0.0001,
    )
    if method_name in delaunay_methods:
        del mesh_SS  # free memory

    bad_point_mask = mask_lie_algebra & ~mask_state_space

    metrics["in_algebra_not_SS"] = np.count_nonzero(bad_point_mask)
    metrics["in_statespace_not_algebra"] = np.count_nonzero(
        mask_state_space & ~mask_lie_algebra
    )
    metrics["in_both"] = np.count_nonzero(mask_lie_algebra & mask_state_space)
    metrics["in_neither"] = np.count_nonzero(~mask_lie_algebra & ~mask_state_space)
    if np.count_nonzero(mask_lie_algebra | mask_state_space) > 0:
        metrics["jaccard_index"] = metrics["in_both"] / np.count_nonzero(
            mask_lie_algebra | mask_state_space
        )
    else:
        metrics["jaccard_index"] = 0.0

    metrics["theorem_holds"] = (
        metrics["empirical_coverage_SS"] >= metrics["empirical_coverage_LA"]
    )
    metrics.update(val_time_metrics)

    if all(
        key in val_time_metrics
        for key in [
            "dynamics_predict_total",
            "error_total",
            "online_calibration_total",
            "sample_boundary_total",
            "mesh_fitting_total",
        ]
    ):
        metrics["full_algorithm_total_s"] = (
            val_time_metrics["dynamics_predict_total"][0]
            + val_time_metrics["error_total"][0]
            + val_time_metrics["online_calibration_total"][0]
            + val_time_metrics["sample_boundary_total"][0]
            + val_time_metrics["mesh_fitting_total"][0]
        )

    # Print organized metrics summary
    print(f"\n=== METRICS SUMMARY for {method_cfg['name']} ===")
    print(
        f"Empirical Coverage - Lie Algebra: {metrics.get('empirical_coverage_LA', 'N/A'):.4f}"
    )
    print(
        f"Empirical Coverage - State Space:  {metrics.get('empirical_coverage_SS', 'N/A'):.4f}"
    )
    print(f"Volume: {metrics.get('total_volume_mesh', 'N/A'):.6f}")
    if "full_algorithm_total_s" in metrics:
        print(f"Full Algorithm Time: {metrics['full_algorithm_total_s']:.6f} s")

    # Print key timing components
    timing_keys = [
        "dynamics_predict_total",
        "error_total",
        "sample_boundary_total",
        "mesh_fitting_total",
    ]
    print("\nTiming Breakdown:")
    for key in timing_keys:
        if key in metrics and isinstance(metrics[key], list) and len(metrics[key]) > 0:
            print(f"  {key:25s}: {metrics[key][0]:.6f} s")

    print("\nMesh Info:")
    print(f"  Vertices: {metrics.get('mesh_n_vertices', 'N/A')}")
    print(f"  Faces: {metrics.get('mesh_n_faces', 'N/A')}")
    print(f"  Watertight: {metrics.get('mesh_is_watertight', 'N/A')}")
    print("=" * 50)

    # Save metrics using new structure
    metrics_path = method_paths["validation"] / f"{p.stem}_{method_name}.pt"
    torch.save(metrics, metrics_path)
    print(f"Saved metrics to: {metrics_path}")

    return metrics


if __name__ == "__main__":
    import argparse

    # import open3d as o3d  # Not needed for data collection
    from luis_utils.conformal_prediction.base import splitCP
    from luis_utils.gaussians import (
        mahalanobis_distance,
        points_in_confidence_hyperellipsoid,
        sample_confidence_ellipsoid_boundary,
    )

    from luis_utils.plotting import visualize_confidence_rerun
    from run_isaac_sysID import run_isaac_sysID

    # Load systems.yaml to get available confidence levels
    try:
        systems_config = yaml.load(open("scripts/systems.yaml"), Loader=yaml.FullLoader)

        # Extract all confidence levels from all robots
        all_confidence_levels = set()
        for robot_name, robot_config in systems_config.items():
            if "estimated_process_noise" in robot_config:
                all_confidence_levels.update(
                    robot_config["estimated_process_noise"].keys()
                )

        # Convert to sorted list for consistent ordering
        available_confidence_levels = sorted(list(all_confidence_levels))

    except Exception as e:
        print(
            f"Warning: Could not load systems.yaml to determine confidence levels: {e}"
        )
        raise e

    parser = argparse.ArgumentParser(description="Run CLAPS pipeline")
    parser.add_argument(
        "--robot_type",
        type=str,
        default="Isaac_Jetbot",
        help="Entry key in scripts/systems.yaml",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="CLAPS",
        choices=list(methods.keys()),
        help="Method configuration to run",
    )
    parser.add_argument(
        "--confidence-level",
        type=str,
        default="under_confident",
        choices=available_confidence_levels,
        help=f"Confidence level (affects process noise). Available: {available_confidence_levels}",
    )
    parser.add_argument(
        "--run-sysid",
        action="store_true",
        help="Run system identification",
        default=False,
    )
    parser.add_argument(
        "--get-calibration-data",
        action="store_true",
        help="Collect calibration data",
        default=False,
    )
    parser.add_argument(
        "--get-validation-data",
        action="store_true",
        help="Collect validation data",
        default=False,
    )
    parser.add_argument(
        "--run-calibration",
        action="store_true",
        help="Run calibration phase",
        default=False,
    )
    parser.add_argument(
        "--run-validation-mesh",
        action="store_true",
        help="Run validation (mesh pipeline)",
        default=False,
    )
    parser.add_argument(
        "--get-metrics",
        action="store_true",
        help="Aggregate metrics after run",
        default=False,
    )
    parser.add_argument(
        "--boundary-points",
        type=int,
        help="Number of boundary points for mesh generation",
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of validation files to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first validation case (for debugging)",
        default=False,
    )
    parser.add_argument(
        "--preserve-validation",
        action="store_true",
        help="Skip cleaning validation data (for debugging)",
        default=False,
    )
    parser.add_argument(
        "--parallel-validation",
        action="store_true",
        help="Process validation files in parallel (default: sequential for accurate timing)",
        default=False,
    )

    args_cli = parser.parse_args()

    robot_name = args_cli.robot_type
    confidence_level = args_cli.confidence_level
    params = yaml.load(open("scripts/systems.yaml"), Loader=yaml.FullLoader)[robot_name]
    params["decimation"] = int(params["planning_dt"] / params["physics_dt"])

    # Validate that the confidence level exists for this robot
    if "estimated_process_noise" in params:
        available_for_robot = list(params["estimated_process_noise"].keys())
        if confidence_level not in available_for_robot:
            raise ValueError(
                f"Confidence level '{confidence_level}' not available for robot '{robot_name}'. "
                f"Available levels: {available_for_robot}"
            )

        # Replace the nested structure with the selected confidence level
        noise_params = params["estimated_process_noise"][confidence_level]
        params["estimated_process_noise"] = noise_params
        print(f"Using {confidence_level} noise parameters: {noise_params}")
    else:
        print(f"No confidence-specific noise parameters defined for {robot_name}")

    # Get boundary points configuration from CLI
    boundary_points = args_cli.boundary_points
    print(f"Using {boundary_points} boundary points for mesh generation")

    # Set up experiment directory structure
    experiment_dir = get_experiment_dir(robot_name, confidence_level, boundary_points)
    reports_dir = get_reports_dir(robot_name, confidence_level, boundary_points)
    print(f"Experiment will save to: {experiment_dir}")
    print(f"Reports will save to: {reports_dir}")

    # Derive RobotType from YAML (e.g., "UNICYCLE" -> RobotType.UNICYCLE)
    robot_type_enum = RobotType[params["robot_type"]]

    # Choose spaces based on method
    method_cfg = methods[args_cli.method]
    model_space = method_cfg["space"]
    calibration_strategy = method_cfg.get("calibration_strategy", "scaling")

    # System keys derived from config
    gt_system_key = (
        robot_type_enum,
        Q_Space.STATE_SPACE,
    )
    method_system_key = (robot_type_enum, model_space)

    if args_cli.run_sysid:
        run_isaac_sysID(
            robot_name,
            params["physics_dt"],
            params["planning_dt"],
            params["decimation"],
            params["sysID"]["total_time"],
        )

    # System parameters
    if params["simulator"] == "isaac":
        isaac_sys_id = torch.load(
            f"data/{robot_name}/sysID/estimated_properties.pt",
            map_location=torch.get_default_device(),
            weights_only=False,
        )
        inertia_matrix = isaac_sys_id["inertia_matrix"]
        inertia_matrix_estimate = isaac_sys_id["inertia_matrix"]

    elif params["simulator"] == "real_robot":
        # For Real MBot, use estimated parameters from system identification
        inertia_matrix = torch.tensor(
            [
                [1.093, 0.0, 0.0],  # Mass in x direction (kg)
                [0.0, 1.093, 0.0],  # Mass in y direction (kg)
                [
                    0.0,
                    0.0,
                    0.0035,
                ],  # Rotational inertia in z (kg⋅m²) (0.5) 0.5 * m * r^2 (r = 0.08) = 0.0035
            ],
            dtype=torch.float64,
        )
        inertia_matrix_estimate = inertia_matrix
    else:
        raise ValueError(f"Unknown simulator type: {params['simulator']}")
    # Convert acceleration bounds to force bounds by multiplying by inertia
    u_acc_bounds = params["u_acc_bounds"]  # [[ax_min, ax_max], [aw_min, aw_max]]
    u_bounds = [
        [
            bound * inertia_matrix_estimate[0, 0].item() for bound in u_acc_bounds[0]
        ],  # Force bounds for linear
        [
            bound * inertia_matrix_estimate[2, 2].item() for bound in u_acc_bounds[1]
        ],  # Force bounds for angular
    ]

    real_system = ROBOT_QSPACE_TO_FACTORY[gt_system_key](inertia_matrix=inertia_matrix)

    # Estimated Aleatoric Disturbances
    Q_cont: torch.Tensor = torch.diag(
        torch.tensor(
            [
                params["estimated_process_noise"]["linear_variance"],
                params["estimated_process_noise"]["angular_variance"],
            ]
        )
    )

    # Actuation noise setup based on robot type (only needed for simulation-based data collection)
    if params["simulator"] == "isaac":
        if robot_name == "Isaac_Jetbot":
            actuation_noise_variance = torch.diag(torch.tensor([0.005, 0.001])) * (
                params["physics_dt"] * params["decimation"]
            )
            print(f"Actuation noise variance: {actuation_noise_variance}")
        else:
            raise ValueError(
                f"Unknown robot type for simulator {params['simulator']}: {robot_name}"
            )

        noise_mvn: torch.distributions.MultivariateNormal = (
            torch.distributions.MultivariateNormal(
                torch.zeros(2), covariance_matrix=actuation_noise_variance
            )
        )
    else:
        # For real robot data, noise variance is not used
        noise_mvn = None

    # Environment setup based on simulator type
    if args_cli.get_calibration_data or args_cli.get_validation_data:
        if params["simulator"] == "isaac":
            print("Using Isaac Lab")
            from isaaclab.app import AppLauncher

            # parser = argparse.ArgumentParser(description="blah blah blah")
            AppLauncher.add_app_launcher_args(parser)
            args_cli = parser.parse_args()
            args_cli.headless = True
            app_launcher = AppLauncher(args_cli)
            simulation_app = app_launcher.app

            from isaac_lab_helpers import IsaacEnv

            real_env = IsaacEnv(
                sim_params={
                    "num_envs": params["max_batch_size_per_env"],
                    "rendering_mode": "performance",
                    "device": args_cli.device,
                    "physics_dt": params["physics_dt"],
                    "decimation": params["decimation"],
                    "max_steps_per_episode": 1,
                    "system_model": real_system,
                }
            )

        elif params["simulator"] == "real_robot":
            # Real robot data is pre-collected, no environment needed for data collection
            print("Using pre-collected real robot data")
            real_env = None  # No environment needed for real robot
        else:
            raise ValueError(
                f"Unknown simulator type: {params['simulator']}. "
                f"Supported types: 'isaac', 'real_robot'"
            )

    method_system = ROBOT_QSPACE_TO_FACTORY[method_system_key](
        inertia_matrix=inertia_matrix_estimate,
    )

    method_env = SecondOrderEnv(
        system=method_system,
        physics_dt=params["physics_dt"],
        max_steps_per_episode=1,
        decimation=params["decimation"],
        integrator_name=params["planner_integrator"],
    )

    approximate_propagator = EKFEstimator(
        system=method_system,
        dt=params["physics_dt"],
        integrator=method_env.integrator,
        error_form=params["error_form"],
    )

    # Debug info (commented out to reduce verbosity)
    print("gt_system_key", gt_system_key)
    print("method_system_key", method_system_key)
    print("method_system", method_system)
    print("method_env", method_env)
    print("approximate_propagator", approximate_propagator)
    print("u_bounds", u_bounds)

    if args_cli.get_calibration_data:
        if params["simulator"] == "real_robot":
            print(
                "❌ Cannot collect calibration data from real robot - data should be pre-processed"
            )
            print(
                "💡 Use process_real_mbot_data.py to convert LCM logs to CLAPS format"
            )
            raise ValueError(
                "Cannot collect calibration data from real robot - data should be pre-processed"
            )
        print("Getting calibration data")
        run_calibration(
            Namespace(
                **{
                    "cal_vel_bounds": params["vel_bounds"],
                    "act_bounds": u_bounds,
                    "Ncal": params["calibration"]["particles_per_grid_point"],
                    "batch": params["max_batch_size_per_env"],
                    "noise_mvn": noise_mvn,
                    "n_vx": params["calibration"]["n_vx"],
                    "n_wz": params["calibration"]["n_wz"],
                    "n_ax": params["calibration"]["n_ax"],
                    "n_aw": params["calibration"]["n_aw"],
                    "env": real_env,
                    "sys_tag": params["simulator"],
                    "robot_name": robot_name,
                }
            )
        )
        pattern = "cal_{}*.pt".format(params["simulator"])
        cal_data_dir = Path(f"data/{robot_name}/raw_data/calibration")
        num_files = len(list(cal_data_dir.glob(pattern)))
        cal_points = num_files * params["calibration"]["particles_per_grid_point"]
        print(f"Cal Data Size: {cal_points}")

    # 1. Conformal Calibration Phase ---------------------
    if args_cli.run_calibration and method_cfg["calibrate"]:
        print("Running Calibration")
        D_cal = {"q0": [], "q1": [], "dq0": [], "dq1": [], "u": []}

        for file in tqdm(
            Path(f"data/{robot_name}/raw_data/calibration").glob(
                f"cal_{params['simulator']}*.pt"
            )
        ):
            loaded_cal_data = torch.load(
                file, map_location=torch.get_default_device(), weights_only=False
            )
            dcal_q0_modelForm = ConfigMapper.map(
                from_key=gt_system_key,
                to_key=method_system_key,
                q=loaded_cal_data["q0"],
            )
            dcal_q1_modelForm = ConfigMapper.map(
                from_key=gt_system_key,
                to_key=method_system_key,
                q=loaded_cal_data["q1"],
            )
            dcal_dq0_modelForm = VelocityMapper.map(
                from_key=gt_system_key,
                to_key=method_system_key,
                q=loaded_cal_data["q0"],
                dq_or_v=loaded_cal_data["dq0"],
            )
            dcal_dq1_modelForm = VelocityMapper.map(
                from_key=gt_system_key,
                to_key=method_system_key,
                q=loaded_cal_data["q1"],
                dq_or_v=loaded_cal_data["dq1"],
            )
            D_cal["q0"].append(dcal_q0_modelForm)
            D_cal["q1"].append(dcal_q1_modelForm)
            D_cal["dq0"].append(dcal_dq0_modelForm)
            D_cal["dq1"].append(dcal_dq1_modelForm)

            # Convert Real_MBot accelerations to forces for consistent dynamics
            u_data = loaded_cal_data["u"].clone()
            if params["simulator"] == "real_robot":
                # Real_MBot data contains accelerations - convert to forces
                u_data[:, 0] *= inertia_matrix[0, 0]  # linear acceleration to force
                u_data[:, 1] *= inertia_matrix[2, 2]  # angular acceleration to torque

            D_cal["u"].append(u_data)

        D_cal["q0"] = torch.cat(D_cal["q0"], dim=0)
        D_cal["q1"] = torch.cat(D_cal["q1"], dim=0)
        D_cal["dq0"] = torch.cat(D_cal["dq0"], dim=0)
        D_cal["dq1"] = torch.cat(D_cal["dq1"], dim=0)
        D_cal["u"] = torch.cat(D_cal["u"], dim=0)

        cal_time_metrics = {}
        with record_time(cal_time_metrics, "calibration_total"):
            # Initial State Estimate
            s_estimate = {
                "P": torch.zeros(D_cal["q0"].shape[0], 6, 6),
                "pose": D_cal["q0"],
                "velocity": D_cal["dq0"],
                "Q": Q_cont,
            }
            s_estimate = approximate_propagator.predict_planning_step(
                s_estimate, D_cal["u"], params["decimation"]
            )

            conf1_hat = s_estimate["pose"]
            xi_sigma_approx = s_estimate["P"][:, :3, :3]

            # Add regularization for numerical stability in state-space methods
            if method_cfg["space"] == Q_Space.STATE_SPACE:
                min_eig = torch.linalg.eigvals(xi_sigma_approx[0]).real.min()
                if min_eig < 1e-5:
                    xi_sigma_approx = (
                        xi_sigma_approx
                        + torch.eye(3, device=xi_sigma_approx.device) * 1e-5
                    )

            if method_cfg["space"] == Q_Space.LIE:
                if params["error_form"] == "RI":
                    err = SE2.log(
                        SE2.right_invariant_error(
                            true_state=D_cal["q1"], estimated_state=conf1_hat
                        )
                    )
                elif params["error_form"] == "LI":
                    err = SE2.right_minus(g_start=conf1_hat, g_end=D_cal["q1"])
            elif method_cfg["space"] == Q_Space.STATE_SPACE:
                err = D_cal["q1"] - conf1_hat

            calibration_strategy = method_cfg.get(
                "calibration_strategy", "scaling"
            )
            scaling_factor: float | None = None
            xi_sigma_calibrated: torch.Tensor | None = None
            bias: torch.Tensor | None = None

            if calibration_strategy == "scaling":
                if method_cfg["r_metric"] == NonconformityScore.MAHALANOBIS:
                    scores = mahalanobis_distance(
                        difference=err, covariance=xi_sigma_approx
                    )
                elif method_cfg["r_metric"] == NonconformityScore.L2:
                    scores = _l2_norm(err, dim=-1)
                else:
                    raise ValueError(
                        f"Invalid r_metric for scaling calibration: {method_cfg['r_metric']}"
                    )

                q_hat = splitCP(scores=scores, alpha=params["failure_rate"])

                if method_cfg["r_metric"] == NonconformityScore.MAHALANOBIS:
                    scaling_factor = (
                        (q_hat**2)
                        / (
                            critical_mahalanobis_distance(
                                failure_rate=params["failure_rate"],
                                D=method_system.DOF,
                            )
                            ** 2
                        )
                    ).item()
                    xi_sigma_calibrated = xi_sigma_approx * scaling_factor
                elif method_cfg["r_metric"] == NonconformityScore.L2:
                    scaling_factor = q_hat.item()  # Critical L2 distance
                else:
                    raise ValueError(
                        f"Invalid r_metric for scaling calibration: {method_cfg['r_metric']}"
                    )

            elif calibration_strategy == "covariance_fit":
                if method_cfg["r_metric"] != NonconformityScore.MAHALANOBIS:
                    raise ValueError(
                        "Covariance fit calibration is only implemented for Mahalanobis metric"
                    )
                xi_sigma_calibrated = estimate_invariant_error_covariance(err)
            elif calibration_strategy == "mean_covariance_fit":
                if method_cfg["r_metric"] != NonconformityScore.MAHALANOBIS:
                    raise ValueError(
                        "Mean+covariance fit calibration is only implemented for Mahalanobis metric"
                    )
                bias, xi_sigma_calibrated = estimate_mean_and_covariance(err)
            else:
                raise ValueError(
                    f"Unsupported calibration strategy: {calibration_strategy}"
                )

        print(f"Calibration Time (s): {cal_time_metrics['calibration_total'][0]:.6f}")

        if method_cfg["r_metric"] == NonconformityScore.MAHALANOBIS:
            if xi_sigma_calibrated is None:
                raise ValueError(
                    "Mahalanobis calibration requires a calibrated covariance"
                )
            errs_for_coverage = err if bias is None else err - bias
            inside_mask, empirical_coverage = points_in_confidence_hyperellipsoid(
                difference=errs_for_coverage,
                covariance=xi_sigma_calibrated,
                failure_rate=params["failure_rate"],
            )
        elif method_cfg["r_metric"] == NonconformityScore.L2:
            distances = _l2_norm(err, dim=-1)
            inside_mask = distances <= scaling_factor
            empirical_coverage = inside_mask.float().mean().item()
        else:
            raise ValueError(f"Invalid r_metric: {method_cfg['r_metric']}")

        print(f"Model: {method_cfg['name']}")
        print(f"Calibrated Coverage: {empirical_coverage}")
        if scaling_factor is not None:
            print(f"Scaling Factor: {scaling_factor}")

        method_paths = ensure_dir_structure(experiment_dir, args_cli.method)

        if calibration_strategy == "covariance_fit":
            cov_path = method_paths["calibration"] / "xi_sigma_calibrated.pt"
            torch.save({"xi_sigma_calibrated": xi_sigma_calibrated.cpu()}, cov_path)
            print(f"Saved calibrated covariance to: {cov_path}")
        elif calibration_strategy == "mean_covariance_fit":
            mean_cov_path = method_paths["calibration"] / "mean_covariance.pt"
            torch.save(
                {
                    "bias": bias.cpu(),
                    "xi_sigma_calibrated": xi_sigma_calibrated.cpu(),
                },
                mean_cov_path,
            )
            print(f"Saved mean bias and calibrated covariance to: {mean_cov_path}")
        else:
            scaling_factor_path = method_paths["calibration"] / "scaling_factor.pt"
            torch.save(
                {"scaling_factor": scaling_factor},
                scaling_factor_path,
            )
            print(f"Saved scaling factor to: {scaling_factor_path}")

    # 2. Validation Phase ---------------------
    if args_cli.get_validation_data:  # (Once per Robot Type)
        if params["simulator"] == "real_robot":
            print(
                "❌ Cannot collect validation data from real robot - data should be pre-processed"
            )
            print(
                "💡 Use process_real_mbot_data.py to convert LCM logs to CLAPS format"
            )
            raise ValueError(
                "Cannot collect validation data from real robot - data should be pre-processed"
            )
        run_validation(
            Namespace(
                **{
                    "val_vel_bounds": params["vel_bounds"],
                    "act_bounds": u_bounds,
                    "Nval": params["validation"]["MC_samples_for_empirical_coverage"],
                    "batch": params["max_batch_size_per_env"],
                    "noise_mvn": noise_mvn,
                    "n_vx": params["validation"]["n_vx"],
                    "n_wz": params["validation"]["n_wz"],
                    "n_ax": params["validation"]["n_ax"],
                    "n_aw": params["validation"]["n_aw"],
                    "env": real_env,
                    "sys_tag": params["simulator"],
                    "robot_name": robot_name,
                }
            )
        )

    if args_cli.run_validation_mesh:
        # Clean previous validation data by default
        if not args_cli.preserve_validation:
            clean_validation_data(experiment_dir, args_cli.method)

        # Ensure directory structure for all methods (needed for validation results)
        method_paths = ensure_dir_structure(experiment_dir, args_cli.method)

        loaded_bias = None
        if not method_cfg[
            "calibrate"
        ]:  # Skip if model does not need calibration (i.e. uncalibrated)
            loaded_scaling_factor = 1.0
            loaded_covariance = None
        elif calibration_strategy == "covariance_fit":
            cov_path = method_paths["calibration"] / "xi_sigma_calibrated.pt"
            loaded = torch.load(
                cov_path,
                map_location=torch.get_default_device(),
                weights_only=False,
            )
            loaded_covariance = loaded["xi_sigma_calibrated"]
            loaded_scaling_factor = None
        elif calibration_strategy == "mean_covariance_fit":
            mean_cov_path = method_paths["calibration"] / "mean_covariance.pt"
            loaded = torch.load(
                mean_cov_path,
                map_location=torch.get_default_device(),
                weights_only=False,
            )
            loaded_bias = loaded["bias"]
            loaded_covariance = loaded["xi_sigma_calibrated"]
            loaded_scaling_factor = None
        else:
            # Load scaling factor using new structure
            scaling_factor_path = method_paths["calibration"] / "scaling_factor.pt"
            loaded_scaling_factor: float = torch.load(
                scaling_factor_path,
                map_location=torch.get_default_device(),
                weights_only=False,
            )["scaling_factor"]
            loaded_covariance = None

        if args_cli.dry_run:
            print("🔬 DRY-RUN MODE: Processing only the first validation case")

        # Get all validation files
        validation_files = sorted(
            Path(f"data/{robot_name}/raw_data/validation").glob(
                f"val_{params['simulator']}*.pt"
            )
        )

        if args_cli.dry_run:
            validation_files = validation_files[:1]  # Only process first file
        elif args_cli.limit:
            validation_files = validation_files[:args_cli.limit]

        # Prepare shared parameters for processing function
        # Convert tensors to CPU for pickling (workers will move back to GPU)
        shared_params = {
            "method_cfg": method_cfg,
            "method_paths": method_paths,
            "loaded_scaling_factor": loaded_scaling_factor,
            "loaded_covariance": loaded_covariance,
            "params": params,
            "robot_name": robot_name,
            "boundary_points": boundary_points,
            "gt_system_key": gt_system_key,
            "method_system_key": method_system_key,
            "inertia_matrix": (
                inertia_matrix.cpu()
                if torch.is_tensor(inertia_matrix)
                else inertia_matrix
            ),
            "Q_cont": Q_cont.cpu() if torch.is_tensor(Q_cont) else Q_cont,
            "approximate_propagator_config": {
                "inertia_matrix_estimate": (
                    inertia_matrix_estimate.cpu()
                    if torch.is_tensor(inertia_matrix_estimate)
                    else inertia_matrix_estimate
                ),
                "physics_dt": params["physics_dt"],
                "planner_integrator": params["planner_integrator"],
                "error_form": params["error_form"],
                "decimation": params["decimation"],
            },
            "method_name": args_cli.method,
            "delaunay_methods": delaunay_methods,
            "calibration_strategy": calibration_strategy,
            "loaded_bias": loaded_bias,
        }

        if args_cli.parallel_validation:
            print(
                f"🚀 PARALLEL MODE: Processing {len(validation_files)} validation files using multiprocessing"
            )

            # Set spawn method for CUDA compatibility (required for CUDA)
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                # Already set, which is fine
                pass

            # Use multiprocessing to process files in parallel
            n_workers = max(1, mp.cpu_count() - 2)
            with mp.Pool(processes=n_workers) as pool:
                # Create argument tuples for starmap
                process_args = [(str(f), shared_params) for f in validation_files]

                # Process all files in parallel
                results = pool.starmap(process_single_validation_file, process_args)

            print(
                f"✅ PARALLEL MODE: Completed processing {len(results)} validation files"
            )

        else:
            print(
                f"📊 SEQUENTIAL MODE: Processing {len(validation_files)} validation files for accurate timing"
            )

            # Process files sequentially
            for p in validation_files:
                metrics = process_single_validation_file(str(p), shared_params)

            print(
                f"✅ SEQUENTIAL MODE: Completed processing {len(validation_files)} validation files"
            )
