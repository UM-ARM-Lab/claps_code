from pathlib import Path

import numpy as np
import torch
from luis_utils.algorithms import Algorithm, Q_Space, RobotType, VelocityMapper
from luis_utils.vecops import wrap_angle
from tqdm import tqdm

def _l2_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.linalg.norm(x, dim=dim)


def estimate_invariant_error_covariance(errors: torch.Tensor) -> torch.Tensor:
    """Compute E[err err^T] for invariant pose errors.

    Args:
        errors: Tensor of shape (N, 3) containing Lie algebra errors.

    Returns:
        (3, 3) covariance matrix.
    """
    if errors.ndim != 2 or errors.shape[1] != 3:
        raise ValueError(
            f"Expected errors with shape (N, 3) for covariance estimation, got {errors.shape}"
        )

    outer_products = errors.unsqueeze(-1) @ errors.unsqueeze(-2)
    return outer_products.mean(dim=0)


def estimate_mean_and_covariance(errors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate bias (mean) and centered covariance of Lie-algebra errors.

    Args:
        errors: (N, 3) Lie algebra errors.

    Returns:
        bias: (3,) mean of errors.
        covariance: (3, 3) covariance of mean-centered errors.
    """
    if errors.ndim != 2 or errors.shape[1] != 3:
        raise ValueError(
            f"Expected errors with shape (N, 3) for mean/covariance estimation, got {errors.shape}"
        )

    bias = errors.mean(dim=0)
    centered = errors - bias
    covariance = estimate_invariant_error_covariance(centered)
    return bias, covariance

# Grid Generation #########################################################
def make_grid(
    vel_bounds: np.ndarray,
    act_bounds: np.ndarray,
    n_vx: int,
    n_wz: int,
    n_ax: int,
    n_aw: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (q0_pts, dq0_pts, u_pts) WITHOUT particle replication in State Space form
    q0_pts: (Nq,3); dq0_pts: (Nd,3); u_pts: (Nu,2)
    """
    # 1a) q0 grid at a single (x,y,θ) cell (0,0,0)
    x = y = theta = torch.tensor([0.0])
    q0_raw = torch.stack(torch.meshgrid(x, y, theta, indexing="ij"), -1).reshape(-1, 3)

    # 1b) dq0 in lie coords
    vx = torch.linspace(*vel_bounds[0], n_vx)
    vy = torch.tensor([0.0])
    wz = torch.linspace(*vel_bounds[1], n_wz)
    dq0_raw_lie = torch.stack(torch.meshgrid(vx, vy, wz, indexing="ij"), -1).reshape(
        -1, 3
    )
    dq0_raw = VelocityMapper.map(
        from_key=(RobotType.UNICYCLE, Q_Space.LIE),
        to_key=(RobotType.UNICYCLE, Q_Space.STATE_SPACE),
        q=q0_raw.repeat(dq0_raw_lie.shape[0], 1),
        dq_or_v=dq0_raw_lie,
    )

    # 1c) actions
    ax = torch.linspace(*act_bounds[0], n_ax)
    aw = torch.linspace(*act_bounds[1], n_aw)
    u_raw = torch.stack(torch.meshgrid(ax, aw, indexing="ij"), -1).reshape(-1, 2)

    return q0_raw, dq0_raw, u_raw


# Collect Data #########################################################
def collect_data(
    env,
    q0_grid: torch.Tensor,
    dq0_grid: torch.Tensor,
    u_grid: torch.Tensor,
    particles_per_pt: int,
    batch_size: int,
    noise_sampler,
    out_dir: Path,
    prefix: str,
):
    """
    For each (q0,dq0,u) triple, replicate it `particles_per_pt` times,
    simulate in chunks of size `batch_size`, and save one file per grid-cell:
      out_dir / f"{prefix}_i{iq:03d}_j{ij:03d}.pt"
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    # flatten the cartesian product
    Mq, Md, Mu = len(q0_grid), len(dq0_grid), len(u_grid)
    combos = torch.cartesian_prod(torch.arange(Mq), torch.arange(Md), torch.arange(Mu))
    for cell_idx, (iq, idq, iu) in enumerate(tqdm(combos, desc=prefix)):
        q0 = q0_grid[iq].unsqueeze(0).repeat(particles_per_pt, 1)
        q0[:, 2] = wrap_angle(q0[:, 2])  # Angle between -pi and pi
        dq0 = dq0_grid[idq].unsqueeze(0).repeat(particles_per_pt, 1)
        u = u_grid[iu].unsqueeze(0).repeat(particles_per_pt, 1)
        # sample your macro‐step noise for this cell
        noise = noise_sampler.sample((particles_per_pt,))
        # now roll in batches
        results = {"q1": [], "dq1": [], "noise": noise, "q0": q0, "dq0": dq0, "u": u}
        for b in range(0, particles_per_pt, batch_size):
            be = min(b + batch_size, particles_per_pt)

            actions = u[b:be].T.unsqueeze(0).unsqueeze(0)  # (H, N_EP, A_DIM, N_EGOS)
            algo_for = Algorithm(
                name="algo_for",
                initial_condition={
                    "q": q0[b:be],
                    "dq": dq0[b:be],
                    "q_reference_space": Q_Space.STATE_SPACE,
                },
                planner_name="fixed_action",
                model_system=env.system,
                planner_kwargs={"planned_controls": actions},
            )
            out_for = env.rollout_episode(
                algorithm=algo_for,  # build your planner per‐batch
                q0=q0[b:be],
                dq0=dq0[b:be],
                output_space=Q_Space.STATE_SPACE,
                ep_index=0,
                noise=noise[b:be].view(-1, 1, 2),
            )
            # Use last step instead of step 1 to handle single-step episodes
            results["q1"].append(out_for["q_traj"][:, -1, :])
            results["dq1"].append(out_for["dq_traj"][:, -1, :])
        # concatenate & save
        results["q1"] = torch.cat(results["q1"], 0)
        results["dq1"] = torch.cat(results["dq1"], 0)
        results["q1"][:, 2] = wrap_angle(
            results["q1"][:, 2]
        )  # Angle between -pi and pi
        torch.save(results, out_dir / f"{prefix}_{cell_idx:04d}.pt")


def run_calibration(args):
    q0g, dq0g, ug = make_grid(
        args.cal_vel_bounds, args.act_bounds, args.n_vx, args.n_wz, args.n_ax, args.n_wz
    )
    collect_data(
        env=args.env,
        q0_grid=q0g,
        dq0_grid=dq0g,
        u_grid=ug,
        particles_per_pt=args.Ncal,
        batch_size=args.batch,
        noise_sampler=args.noise_mvn,
        out_dir=Path(f"data/{args.robot_name}/raw_data/calibration"),
        prefix=f"cal_{args.sys_tag}",
    )


def run_validation(args):
    q0g, dq0g, ug = make_grid(
        args.val_vel_bounds, args.act_bounds, args.n_vx, args.n_wz, args.n_ax, args.n_wz
    )
    collect_data(
        env=args.env,
        q0_grid=q0g,
        dq0_grid=dq0g,
        u_grid=ug,
        particles_per_pt=args.Nval,  # e.g. 50k
        batch_size=args.batch,
        noise_sampler=args.noise_mvn,
        out_dir=Path(f"data/{args.robot_name}/raw_data/validation"),
        prefix=f"val_{args.sys_tag}",
    )
