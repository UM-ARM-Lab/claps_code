import torch
from typing import Optional, Tuple
from luis_utils.algorithms import Q_Space, RobotType, ROBOT_TO_KEY, VelocityMapper

try:
    from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv
    from isaaclab.managers import ActionTerm, ActionTermCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass
    from isaaclab.assets.articulation import Articulation
except Exception as e:
    raise ImportError("Please create your Isaac SimulationApp before importing isaac_utils") from e


def reset_base(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    with torch.inference_mode():
        asset = env.scene[asset_cfg.name]
        
        q = env.reset_base_properties["q"]
        dq = env.reset_base_properties["dq"]

        default_state = asset.data.default_root_state.clone()
        x, y, yaw = torch.chunk(q, 3, dim=1)
        z = default_state[:, 2:3]
        quat = quat_from_euler_xyz(torch.zeros_like(yaw.flatten()), torch.zeros_like(yaw.flatten()), yaw.flatten())
        pose = torch.cat([x, y, z, quat], dim=1)
        pose[:, :2] += env.scene.env_origins[:, :2]
        asset.write_root_pose_to_sim(pose)

        # # Velocity
        dq_world = dq
        dq_body = VelocityMapper.map(from_key=env.system_key, to_key=(env.system_key[0], Q_Space.LIE), q=q, dq_or_v=dq)

        lin_vel = torch.cat([dq_world[:, 0:2], torch.zeros_like(dq_world[:, 2:3])], dim=1)
        ang_vel = torch.cat([torch.zeros_like(dq_world[:, 0:2]), dq_world[:, 2:3]], dim=1)
        asset.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=1))

        # Joint State
        starting_joint_vels = dq_body[:, [0, 2]] @ env.action_manager.cfg.test2.velocity_jac_T
        joint_pos, _ = (
            asset.data.default_joint_pos.clone(),
            asset.data.default_joint_vel.clone(),
        )

        asset.write_joint_state_to_sim(joint_pos, starting_joint_vels)


def zero_COM(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    com_pose = asset.root_physx_view.get_coms()
    com_pose[:, :, :3] = 0.0 # zero position
    com_pose[:, :, 3:6] = 0.0 # zero orientation
    com_pose[:, :, 6:7] = 1.0 # w element of quaternion
    asset.root_physx_view.set_coms(com_pose, torch.tensor([x for x in range(com_pose.shape[0])], device=com_pose.device))


class JetbotActionTerm(ActionTerm):

    _asset: Articulation

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # create buffers
        self._raw_actions = torch.zeros(env.num_envs, 2, device=self.device) # a_x, tau_z
        self._processed_actions = torch.zeros(env.num_envs, 2, device=self.device) # wheel_accel_l, wheel_accel_r
        self._torque_command = torch.zeros(env.num_envs, 2, device=self.device) # torque_l, torque_r
        self._disturbed_actions = torch.zeros(env.num_envs, 2, device=self.device) # torque_l, torque_r
        self._R = cfg.R
        self._L = cfg.L
        self.torque_jacobian = cfg.torque_jacobian
        self.velocity_jac_T = cfg.velocity_jac_T
        assert torch.allclose(torch.linalg.inv(self.velocity_jac_T), self.torque_jacobian)

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def process_actions(self, action: torch.Tensor):
        self._raw_actions[:] = action # Actions are body forces/torques (Fx and Tz)
        self._processed_actions[:] = self._raw_actions[:] @ self.torque_jacobian.T

    def apply_actions(self):
        self._torque_command[:] = self._processed_actions # (N, 2)
        self._asset.set_joint_effort_target(self._torque_command)

@configclass
class JetbotActionTermCfg(ActionTermCfg):
    class_type: type = JetbotActionTerm

    R: float = 0.0325 # Wheel radius (m)
    L: float = 0.118 # wheel separation (m)

    torque_jacobian = torch.tensor([[R / 2, - R / L], [R / 2, R / L]])
    velocity_jac_T = torch.tensor([[1/R, - L / (2 * R)], [1/R, L / (2 * R)]]).T

def get_q(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    state = asset.data.root_state_w.clone()
    xy = state[:, :2]
    _, _, yaw = euler_xyz_from_quat(state[:, 3:7])
    xy -= env.scene.env_origins[:, :2]
    return torch.cat([xy, yaw.view(-1, 1)], dim=1) # (N, 3)

def get_dq(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    state = asset.data.root_state_w.clone()
    xy_vel = state[:, 7:9]
    yaw_vel = state[:, 12]
    return torch.cat([xy_vel, yaw_vel.view(-1, 1)], dim=1) # (N, 3)

def get_measured_torques(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_link_incoming_joint_force()[:, 1:3, 3] # (N, 2)

def get_commanded_torques(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return env.action_manager._terms['test2']._torque_command # (N, 2)

def get_wheel_angles(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, 0:2] # (N, 2)

def get_wheel_velocities(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, 0:2] # (N, 2)

class MySceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))

    jetbot: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd"),
        actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=0.0, stiffness=None,
            velocity_limit=1e6, velocity_limit_sim=1e6)},    ).replace(prim_path="{ENV_REGEX_NS}/Jetbot")


@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    test2 = JetbotActionTermCfg(asset_name="jetbot")

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""
    @configclass
    class PolicyCfg(ObsGroup):

        # observation terms (order preserved)
        q = ObsTerm(func=get_q, params={"asset_cfg": SceneEntityCfg("jetbot")})
        dq = ObsTerm(func=get_dq, params={"asset_cfg": SceneEntityCfg("jetbot")})
        measured_torques = ObsTerm(func=get_measured_torques, params={"asset_cfg": SceneEntityCfg("jetbot")})
        commanded_torques = ObsTerm(func=get_commanded_torques, params={"asset_cfg": SceneEntityCfg("jetbot")})
        wheel_angles = ObsTerm(func=get_wheel_angles, params={"asset_cfg": SceneEntityCfg("jetbot")})
        wheel_velocities = ObsTerm(func=get_wheel_velocities, params={"asset_cfg": SceneEntityCfg("jetbot")})
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()



@configclass
class EventCfg:

    reset_base = EventTerm(
        func=reset_base,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("jetbot")}
    )
    zero_COM = EventTerm(
        func=zero_COM,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("jetbot")}
    )


@configclass
class CarEnvCfg(ManagerBasedEnvCfg):

    render: sim_utils.RenderCfg = sim_utils.RenderCfg(rendering_mode="performance")

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.sim.render_interval = 1
        self.viewer.eye = [3.5, 0.0, 3.2]
        self.viewer.lookat = [0.0, 0.0, 0.5]



def create_env(params: dict):
    scene = MySceneCfg(num_envs=params["num_envs"], env_spacing=params["env_spacing"], replicate_physics=params["replicate_physics"])
    render_cfg = sim_utils.RenderCfg(rendering_mode=params["rendering_mode"]) # includes performance, balanced and quality
    physx_cfg = sim_utils.PhysxCfg(solver_type=params["solver_type"])
    sim_cfg = sim_utils.SimulationCfg(device=params["device"], dt=params["physics_dt"], render=render_cfg, physx=physx_cfg)

    env = ManagerBasedEnv(cfg=CarEnvCfg(sim=sim_cfg, decimation=params["decimation"], scene=scene))
    env.seed(params["seed"])
    print(f"sim dt {env.sim.get_physics_dt()}")
    print(f"step_dt {env.step_dt}")
    print(f"decimation {env.cfg.decimation}")
    
    return env

from luis_utils.env import SecondOrderEnv
class IsaacEnv(SecondOrderEnv):

    def __init__(self, sim_params: dict):

        # Set default parameters if not provided
        if "solver_type" not in sim_params:
            sim_params["solver_type"] = 0
        if "replicate_physics" not in sim_params:
            sim_params["replicate_physics"] = True
        if "seed" not in sim_params:
            sim_params["seed"] = 0
        if "env_spacing" not in sim_params:
            sim_params["env_spacing"] = 2.0

                
        self.isaac_env = create_env(sim_params)

        super().__init__(system=sim_params["system_model"], physics_dt=sim_params["physics_dt"], max_steps_per_episode=sim_params["max_steps_per_episode"], integrator_name="isaac-sim",
            decimation=sim_params["decimation"])
        self.isaac_obs = None
        self.isaac_env.system_key = self.system_key # Pass attribute to isaac env class
        self.isaac_env.reset_base_properties = {}
        self.N = sim_params["num_envs"]
        assert self.isaac_env.step_dt == sim_params["physics_dt"] * sim_params["decimation"]

    def _post_reset(self):
        # Perform reset in Isaac Sim
        self.isaac_env.reset_base_properties["q"] = self.q
        self.isaac_env.reset_base_properties["dq"] = self.dq
        self.isaac_obs, _ = self.isaac_env.reset()
        # Set class variables to be those obtained after Isaac Sim reset
        self.q = self.isaac_obs["policy"]["q"]
        self.dq = self.isaac_obs["policy"]["dq"]

    def dynamics_step(self, _q: torch.Tensor, _dq: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode(): # Isaac does decimation internally
            self.isaac_obs, _ = self.isaac_env.step(u)

        return self.isaac_obs['policy']['q'], self.isaac_obs['policy']['dq']

    def get_observation(self) -> dict:
        out = {}
        for key, value in self.isaac_obs['policy'].items(): # TODO: why policy 
            out[key] = value
        return out
