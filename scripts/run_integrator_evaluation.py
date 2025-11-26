import torch
import time
import numpy as np
from luis_utils.load import script_starter

script_starter(precision="double", device="cpu")

from luis_utils.algorithms import (
    ROBOT_QSPACE_TO_FACTORY,
    Algorithm,
    Q_Space,
    RobotType,
    ConfigMapper,
    VelocityMapper,
    ROBOT_TO_KEY,
)
from luis_utils.env import rollout_analytical_trajectory, SecondOrderEnv
from luis_utils.systems.second_order_unicycle import (
    analytical_constant_acceleration_unicycle,
)
from luis_utils.load import get_repo_path, get_callable
from tqdm import tqdm

from pymatlie.se2 import SE2

# Constants ---------------------------------------------------------------
total_time = 1.0
dt_values = np.logspace(np.log10(0.001), np.log10(0.1), 7)
decimation = 1
analytical_system_key = (RobotType.UNICYCLE, Q_Space.STATE_SPACE)
planner_name = "fixed_action"
lie_group_integrators = [
    "cf4",
    "rk2",
    "se",
    "heun",
    "fe",
]
integrator_list = ["fe", "se", "heun", "rk2", "rk4"]
inertia_matrix = torch.load(
    "data/sysID/estimated_properties.pt", map_location=torch.get_default_device()
)["inertia_matrix"]
qref_space = Q_Space.STATE_SPACE
N_time = 50_000

# define grids
# initial velocities (v0, ω0)
v_vals = torch.linspace(0.1, 0.5, 5)
w_vals = torch.linspace(0.0, 0.5, 5)
init_grid = [(float(v), float(w)) for v in v_vals for w in w_vals]

# constant accelerations (a, b)
a_vals = torch.linspace(0.0, 0.5, 5)
b_vals = torch.linspace(0.0, 2.0, 5)
ctrl_grid = [(float(a), float(b)) for a in a_vals for b in b_vals]

real_systems = [
    [
        ROBOT_QSPACE_TO_FACTORY[(analytical_system_key[0], Q_Space.STATE_SPACE)](
            inertia_matrix
        ),
        "State-Space",
    ],  # Conventional State-Space representation
    [
        ROBOT_QSPACE_TO_FACTORY[(analytical_system_key[0], Q_Space.LIE)](
            inertia_matrix
        ),
        "Euler-Poincare-Suslov",
    ],  # Lie Group Representation using EPS
]

combos = [
    (float(v), float(w), float(a), float(b))
    for v in v_vals
    for w in w_vals
    for a in a_vals
    for b in b_vals
]
print(f"Combos: {len(combos)}")
rows = []

state_metrics = {
    name: {"q_errors": {}, "total_time": [], "int_time_ms": []}
    for name in integrator_list
}
lie_metrics = {
    name: {"q_errors": {}, "total_time": [], "int_time_ms": []}
    for name in lie_group_integrators
}

for dt in tqdm(dt_values, desc="dt sweep"):
    # build time array once
    num_steps = int(total_time / dt)
    times = torch.linspace(
        0,
        dt * num_steps,
        num_steps + 1,
        dtype=torch.double,
        device=torch.get_default_device(),
    )

    v0_dummy, w0_dummy = init_grid[0]
    t0_u_dummy = torch.tensor(
        [[v0_dummy, w0_dummy]], dtype=torch.double, device=torch.get_default_device()
    )
    t0_q0_raw = torch.tensor(
        [[0.0, 0.0, 0.0]], dtype=torch.double, device=torch.get_default_device()
    )
    t0_dx0_dummy = v0_dummy * np.cos(0.0)
    t0_dy0_dummy = v0_dummy * np.sin(0.0)
    t0_dq0_raw = torch.tensor(
        [[t0_dx0_dummy, t0_dy0_dummy, w0_dummy]],
        dtype=torch.double,
        device=torch.get_default_device(),
    )

    for system, sys_name in real_systems:
        is_lie = sys_name == "Euler-Poincare-Suslov"
        system_key = ROBOT_TO_KEY[system.__class__]

        t0_q_dummy = ConfigMapper.map(analytical_system_key, system_key, t0_q0_raw)
        t0_dq_dummy = VelocityMapper.map(
            analytical_system_key, system_key, t0_q0_raw, t0_dq0_raw
        )

        int_list = lie_group_integrators if is_lie else integrator_list

        for integ in tqdm(int_list, desc=f"{sys_name} ints", leave=False):
            fn = get_callable(f"luis_utils.integrators", integ)

            # warm up (jit, cache, etc)
            for _ in range(10):
                _ = fn(system, t0_q_dummy, t0_dq_dummy, t0_u_dummy, dt)
            # torch.cuda.synchronize()
            # time N back‐to‐back calls
            t0 = time.perf_counter()
            for _ in range(N_time):
                _ = fn(system, t0_q_dummy, t0_dq_dummy, t0_u_dummy, dt)
            # torch.cuda.synchronize()
            avg_int_ms = (time.perf_counter() - t0) / N_time * 1e3
            target = lie_metrics if is_lie else state_metrics
            target[integ]["int_time_ms"].append(avg_int_ms)

            errs = []
            tms = []

            for v0, w0 in init_grid:
                q0_raw = torch.tensor(
                    [[0.0, 0.0, 0.0]],
                    dtype=torch.double,
                    device=torch.get_default_device(),
                )
                dx0 = v0 * np.cos(0.0)
                dy0 = v0 * np.sin(0.0)
                dq0_raw = torch.tensor(
                    [[dx0, dy0, w0]],
                    dtype=torch.double,
                    device=torch.get_default_device(),
                )

                q0 = ConfigMapper.map(analytical_system_key, system_key, q0_raw)
                dq0 = VelocityMapper.map(
                    analytical_system_key, system_key, q0_raw, dq0_raw
                )

                for a, b in ctrl_grid:

                    u_force = torch.tensor(
                        [
                            [a * inertia_matrix[0, 0].item()],
                            [b * inertia_matrix[2, 2].item()],
                        ]
                    ).T  # (1,2)
                    algo = Algorithm(
                        name=f"{sys_name}_{integ}_dt{dt}",
                        initial_condition={
                            "q": q0,
                            "dq": dq0,
                            "q_reference_space": qref_space,
                        },
                        planner_name=planner_name,
                        planner_kwargs={"planned_controls": u_force},
                        model_system=ROBOT_QSPACE_TO_FACTORY[analytical_system_key](
                            inertia_matrix
                        ),
                    )
                    env = SecondOrderEnv(system, dt, num_steps, integ, decimation)
                    out = env.rollout_episode(
                        algo,
                        q0=q0,
                        dq0=dq0,
                        output_space=Q_Space.STATE_SPACE,
                        noise=None,
                    )

                    # analytic
                    ref = rollout_analytical_trajectory(
                        out,
                        inertia_matrix,
                        q0=q0_raw,
                        dq0=dq0_raw,
                        u=torch.tensor([[a, b]]),
                        analytical_func=analytical_constant_acceleration_unicycle,
                    )

                    err_sq = []
                    for t in range(num_steps):
                        g_sim_final = ConfigMapper.map(
                            analytical_system_key,
                            (analytical_system_key[0], Q_Space.LIE),
                            out["q_traj"][:, t + 1, :],
                        )
                        g_ref_final = ConfigMapper.map(
                            analytical_system_key,
                            (analytical_system_key[0], Q_Space.LIE),
                            ref["q_traj"][:, t + 1, :],
                        )
                        err = SE2.right_minus(
                            g_start=g_sim_final, g_end=g_ref_final
                        )  # does log(g_start^{-1} @ g_end)
                        err_sq.append(err.norm(dim=-1).pow(2).item())

                    errs.append(np.sqrt(np.mean(err_sq)))

            mean_err = np.mean(errs)

            target = lie_metrics if is_lie else state_metrics
            target[integ]["q_errors"][dt] = mean_err

torch.save(
    {"state_metrics": state_metrics, "lie_metrics": lie_metrics},
    "data/analytical/integrator_metrics.pt",
)
