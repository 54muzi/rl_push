"""Minimal fixed-base G1 environment for asset and task-registration debugging.
python scripts/train.py --task Debug-G1 --num_envs 2 --max_iterations 100000  --device cuda:0

运行后会打开 Debug-G1 Joint Controller 窗口：

  - 每个关节有 position slider。
  - 每个关节有 P / D gain slider。
  - 右侧 effort bar 显示当前关节力矩。
  - Reset 回默认姿态。
  - Randomize 在 soft joint limits 内随机姿态。
  - policy action 会被忽略，GUI slider 是实际控制源。
  - num_envs=2 时，env0 用原始目标，env1 用左右镜像目标；如果只想看一个机器人，
    用 --num_envs 1。


无窗口 smoke test 用：
G1_DEBUG_GUI=0 python scripts/train.py --task Debug-G1 --num_envs 2  --max_iterations 1 --headless

"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from g1_loco_manipulation import mdp
from g1_loco_manipulation.assets.robots.g1 import UNITREE_G1_29DOF_CFG as ROBOT_CFG


DEBUG_PLANE_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(4.0, 4.0),
    border_width=4.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)


@configclass
class DebugSceneCfg(InteractiveSceneCfg):
    """Flat scene containing only the G1 robot and contact sensors."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=DEBUG_PLANE_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


@configclass
class CommandsCfg:
    """No command terms are needed for the fixed-base debug task."""


@configclass
class ActionsCfg:
    """GUI-controlled joint-position actions around the robot default pose."""

    joint_pos = mdp.JointPositionGUIActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-1.0, 1.0)},
        mirror_actions=True,
    )


@configclass
class ObservationsCfg:
    """Small policy observation set for RSL-RL smoke tests."""

    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Dummy reward for keeping the RSL-RL loop valid."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class TerminationsCfg:
    """Only time-out terminates the fixed-base debug episode."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """Deterministic reset to the robot default pose."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "z": (0.8, 0.8),
                "yaw": (math.pi / 2, math.pi / 2),
            },
            "velocity_range": {},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class G1DebugEnvCfg(ManagerBasedRLEnvCfg):
    """Fixed-base G1 debug environment."""

    scene: DebugSceneCfg = DebugSceneCfg(num_envs=2, env_spacing=1.25)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 3600.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.fix_base = True
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.scene.contact_forces.update_period = self.sim.dt
