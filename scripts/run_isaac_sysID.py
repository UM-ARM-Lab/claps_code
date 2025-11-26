import torch

from luis_utils.algorithms import Algorithm, Q_Space, ROBOT_QSPACE_TO_FACTORY, RobotType, VelocityMapper
import numpy as np
import matplotlib.pyplot as plt

def estimate_mass(env, forward_forces, robot_name):

    # Start from rest at origin
    q0 = torch.zeros(env.N, 3, device=env.isaac_env.device)
    dq0 = torch.zeros_like(q0)

    m_means = []
    a_means = []
    radius_means = []
    for fx in forward_forces:
        planned_controls = torch.tensor([fx, 0.0], device=env.isaac_env.device).view(1, 2).repeat(env.N, 1)

        algo = Algorithm(
            name="fixed_action_algo",
            initial_condition={"q": q0, "dq": dq0, "q_reference_space": env.system_key[1]},
            planner_name="fixed_action",
            planner_kwargs={"planned_controls": planned_controls.T.unsqueeze(0).unsqueeze(0)},
            model_system=env.system,
        )

        out_dic = env.rollout_episode(algorithm=algo, q0=q0, dq0=dq0, output_space=Q_Space.STATE_SPACE, noise=None, ep_index=0)

        q_traj = out_dic["q_traj"]
        dq_traj = out_dic["dq_traj"]
        T_steps = q_traj.shape[1]
        
        dq_body = VelocityMapper.map(from_key=(RobotType.UNICYCLE, Q_Space.STATE_SPACE), to_key=(RobotType.UNICYCLE, Q_Space.LIE), q=q_traj.reshape(-1, 3), dq_or_v=dq_traj.reshape(-1, 3))

        dq_body = dq_body.reshape(env.N, T_steps, 3) # (N, T, 3)
        vx_body = dq_body[:, :, 0]

        t = (torch.arange(T_steps, device=env.isaac_env.device) * env.isaac_env.step_dt).cpu().numpy()
        vx_mean = vx_body.mean(dim=0).cpu().numpy()
        slope_mean, intercept_mean = np.polyfit(t, vx_mean, 1)
        m_i = fx / slope_mean

        vx_body = vx_body.cpu()
        q_traj = q_traj.cpu()

        wheel_vel_left = out_dic["wheel_velocities_traj"][:, :, 0].cpu()
        wheel_vel_right = out_dic["wheel_velocities_traj"][:, :, 1].cpu()

        sum_w = (wheel_vel_left + wheel_vel_right) / 2.0
        mask = sum_w > 1e-6

        radius_est, _ = np.polyfit(sum_w[mask], vx_body[mask], 1)
        radius_means.append(radius_est)
        plt.figure()
        plt.plot(sum_w[mask], vx_body[mask], 'o', label="data")
        plt.plot(sum_w[mask], radius_est * sum_w[mask], label="fit")
        plt.xlabel("Wheel velocity (rad/s)")
        plt.ylabel("Velocity $v_x$ (m/s)")
        plt.title(f"Mean fit: radius={radius_est:.5f} m")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"data/{robot_name}/sysID/debug_plots/traj_vel_plot_Fx{fx}_radius_est.png")

        m_per_robot = []
        for rob_i in range(env.N):
            slope_i, _ = np.polyfit(t, vx_body[rob_i], 1)
            m_per_robot.append(fx / slope_i)
        m_per_robot = np.array(m_per_robot)
        plt.figure()
        plt.errorbar(t, vx_mean, yerr=vx_body.std(dim=0).cpu().numpy(), fmt='o', label="mean ± std")
        plt.plot(t, slope_mean * t + intercept_mean, label="linear fit")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity $v_x$ (m/s)")
        plt.title(f"Mean fit: a={slope_mean:.5f} m/s², m_est={m_i:.5f} kg")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"data/{robot_name}/sysID/debug_plots/traj_vel_plot_Fx{fx}.png")
        m_means.append(m_i)
        a_means.append(slope_mean)

        plt.figure()
        plt.hist(m_per_robot, bins=20)
        plt.xlabel("Estimated mass per robot (kg)")
        plt.ylabel("Count")
        plt.title("Distribution of per-robot mass estimates")
        plt.tight_layout()
        plt.savefig(f"data/{robot_name}/sysID/debug_plots/traj_mass_distribution_Fx{fx}.png")

        plt.figure()
        plt.plot(q_traj[:, :, 0].T, q_traj[:, :, 1].T)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("XY trajectories")
        plt.axis("equal")
        plt.savefig(f"data/{robot_name}/sysID/debug_plots/traj_xy_plot_Fx{fx}.png")

    return m_means, a_means, radius_means

def estimate_inertia(env, torques_z, robot_name):
    q0 = torch.zeros(env.N, 3, device=env.isaac_env.device)
    dq0 = torch.zeros_like(q0)

    I_estimates = []
    α_estimates = []
    for τ in torques_z:
        planned_controls = torch.tensor([0.0, τ], device=env.isaac_env.device) \
                             .view(1,2).repeat(env.N,1)
        
        algo = Algorithm(
            name="fixed_action_algo",
            initial_condition={"q": q0, "dq": dq0, "q_reference_space": env.system_key[1]},
            planner_name="fixed_action",
            planner_kwargs={"planned_controls": planned_controls.T.unsqueeze(0).unsqueeze(0)},
            model_system=env.system,
        )

        out_dic = env.rollout_episode(algorithm=algo, q0=q0, dq0=dq0, output_space=Q_Space.STATE_SPACE, noise=None, ep_index=0)

        q_traj = out_dic["q_traj"]
        dq_traj = out_dic["dq_traj"]
        T_steps = q_traj.shape[1]

        dq_body = VelocityMapper.map(from_key=(RobotType.UNICYCLE, Q_Space.STATE_SPACE), to_key=(RobotType.UNICYCLE, Q_Space.LIE), q=q_traj.reshape(-1, 3), dq_or_v=dq_traj.reshape(-1, 3))

        dq_body = dq_body.reshape(env.N, T_steps, 3) # (N, T, 3)
        wz = dq_body[:,:,2].cpu().numpy()
        t  = np.arange(wz.shape[1]) * env.isaac_env.step_dt

        wz_mean = wz.mean(axis=0)
        α, intercept = np.polyfit(t, wz_mean, 1)
        I_estimates.append(τ / α)
        α_estimates.append(α)

        plt.figure()
        plt.errorbar(t, wz_mean, yerr=wz.std(axis=0), fmt='o', label="mean ± std")
        plt.plot(t, α * t + intercept, label="linear fit")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular velocity $w_z$ (rad/s)")
        plt.title(f"Mean fit: α={α:.5f} rad/s², I_est={τ / α:.5f} kg·m²")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"data/{robot_name}/sysID/debug_plots/traj_ang_vel_plot_τ{τ}.png")

    return I_estimates, α_estimates


def validate_identified_params(
    env,
    test_cmds: np.ndarray,   # shape (K,2) rows: [Fx, tau_z]
    m_hat: float,
    I_hat: float,
    robot_name: str,
    linear_window: float = 0.5,  # seconds of data to fit
):
    q0 = torch.zeros(env.N, 3, device=env.isaac_env.device)
    dq0 = torch.zeros_like(q0)
    dt = env.isaac_env.step_dt

    results = []
    for Fx, tau_z in test_cmds:
        # 1) simulate
        controls = torch.tensor([Fx, tau_z], device=env.isaac_env.device)\
                          .view(1,2).repeat(env.N,1)
        algo = Algorithm(
            name="val",
            initial_condition={
              "q": q0, "dq": dq0,
              "q_reference_space": env.system_key[1]
            },
            planner_name="fixed_action",
            planner_kwargs={"planned_controls": controls.T.unsqueeze(0).unsqueeze(0)},
            model_system=env.system,
        )
        out = env.rollout_episode(
            algorithm=algo, q0=q0, dq0=dq0,
            output_space=Q_Space.STATE_SPACE,
            noise=None,
            ep_index=0
        )

        # 2) get actual T and times
        N, T, _ = out["q_traj"].shape
        times = np.arange(T) * dt

        # 3) extract body‐frame v & ω in one shot
        flat_v = VelocityMapper.map(
            from_key=(RobotType.UNICYCLE, Q_Space.STATE_SPACE),
            to_key=(RobotType.UNICYCLE, Q_Space.LIE),
            q=out["q_traj"].reshape(-1,3),
            dq_or_v=out["dq_traj"].reshape(-1,3)
        )  # shape (N*T, 3)
        traj_v = flat_v.reshape(N, T, 3).cpu().numpy()

        vx = traj_v[:,:,0].mean(axis=0)   # (T,)
        wz = traj_v[:,:,2].mean(axis=0)   # (T,)

        # 4) only fit initial 'linear_window' seconds
        mask = times <= linear_window

        a_sim,     _ = np.polyfit(times[mask], vx[mask], 1)
        alpha_sim, _ = np.polyfit(times[mask], wz[mask], 1)

        a_pred     = Fx    / m_hat
        alpha_pred = tau_z / I_hat

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
        ax1.plot(times, vx,           label="vₓ (sim)")
        ax1.plot(times, a_pred*times, '--', label=f"vₓ (pred={a_pred:.3f}·t)")
        ax1.set_xlabel("t [s]"); ax1.set_ylabel("vₓ [m/s]")
        ax1.legend()
        ax2.plot(times, wz,              label="ω_z (sim)")
        ax2.plot(times, alpha_pred*times, '--', label=f"ω_z (pred={alpha_pred:.3f}·t)")
        ax2.set_xlabel("t [s]"); ax2.set_ylabel("ω_z [rad/s]")
        ax2.legend()

        plt.suptitle(f"Cmd Fx={Fx:.2f}, τ={tau_z:.3f}")
        plt.tight_layout()
        plt.savefig(f"data/{robot_name}/sysID/debug_plots/validate_F{Fx:.2f}_τ{tau_z:.3f}.png")
        plt.close(fig)

        results.append({
            "Fx": Fx,
            "tau_z": tau_z,
            "a_sim": a_sim,
            "a_pred": a_pred,
            "alpha_sim": alpha_sim,
            "alpha_pred": alpha_pred,
            "rel_err_a": abs(a_sim - a_pred)/a_pred if a_pred != 0 else 0,
            "rel_err_alpha": abs(alpha_sim - alpha_pred)/alpha_pred if alpha_pred != 0 else 0,
        })

    return results



def run_isaac_sysID(robot_name, physics_dt, planning_dt, decimation, total_time):

    from luis_utils.load import script_starter
    script_starter(precision="double", device="cuda")

    from isaaclab.app import AppLauncher
    import argparse
    parser = argparse.ArgumentParser(description="blah blah blah")
    parser.add_argument("--num_envs", type=int, default=300, help="Number of environments to spawn.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from isaac_lab_helpers import IsaacEnv

    inertia_matrix = torch.diag(torch.tensor([200.0, 200.0, 200.0]))
    isaac_sim_key = (RobotType.UNICYCLE, Q_Space.STATE_SPACE)
    isaac_system = ROBOT_QSPACE_TO_FACTORY[isaac_sim_key](inertia_matrix)
    isaac_env = IsaacEnv(sim_params={
        "num_envs": 300, "rendering_mode": "quality", "device": "cuda", "physics_dt": physics_dt,
        "decimation": decimation,
        "max_steps_per_episode": int(total_time / planning_dt),
        "system_model": isaac_system,
        "noise_reference_key": (isaac_sim_key[0], Q_Space.LIE)
    })

    mass_estimates, accel_estimates, radius_estimates = estimate_mass(isaac_env, forward_forces=np.array([0.1, 0.2, 0.3, 0.4]), robot_name=robot_name)
    print(f"Estimated masses: {mass_estimates}; mean: {np.mean(mass_estimates)}")
    print(f"Estimated Forces: {np.array(accel_estimates) * np.mean(mass_estimates)}")
    print(f"Estimated radii: {radius_estimates}; mean: {np.mean(radius_estimates)}")

    inertia_estimates, α_estimates = estimate_inertia(isaac_env, torques_z=np.array([0.01, 0.015, 0.02, 0.025]), robot_name=robot_name)
    print(f"Estimated inertias: {inertia_estimates}; mean: {np.mean(inertia_estimates)}")
    print(f"Estimated torques: {np.array(α_estimates) * np.mean(inertia_estimates)}")


    isaac_sys_id  = {"estimated_mass": np.mean(mass_estimates), "estimated_inertia": np.mean(inertia_estimates),
            "inertia_matrix": torch.diag(torch.tensor([np.mean(mass_estimates), np.mean(mass_estimates), np.mean(inertia_estimates)]))}
    torch.save(isaac_sys_id, f"data/{robot_name}/sysID/estimated_properties.pt")

    errs = validate_identified_params(isaac_env, np.array([
        [0.2, 0.01],
        [0.3, 0.01],
        [0.4, 0.02],
        [0.1, 0.02],
        [0.4, 0.0],
        [0.0, 0.02],
    ]), np.mean(mass_estimates), np.mean(inertia_estimates), robot_name=robot_name)

    for e in errs:
        print(f"Cmd F={e['Fx']:.2f},τ={e['tau_z']:.3f}:  "
            f"a_sim={e['a_sim']:.3f}, a_pred={e['a_pred']:.3f}, "
            f"err_a={100*e['rel_err_a']:.1f}%,  "
            f"α_sim={e['alpha_sim']:.3f}, α_pred={e['alpha_pred']:.3f}, "
            f"err_α={100*e['rel_err_alpha']:.1f}%")

    isaac_env.isaac_env.close()