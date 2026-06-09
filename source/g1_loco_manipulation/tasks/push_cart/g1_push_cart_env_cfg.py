import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from g1_loco_manipulation.assets.objects.service_cart import SERVICE_CART_V1_CFG
from g1_loco_manipulation.assets.robots.g1 import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from g1_loco_manipulation.tasks.push_cart import mdp


SIMPLE_PLANE_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(6.0, 4.0),
    border_width=8.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)


@configclass
class PushCartSceneCfg(InteractiveSceneCfg):
    """Flat-ground scene containing G1 and the rail-constrained service cart."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SIMPLE_PLANE_CFG,
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
    cart: ArticulationCfg = SERVICE_CART_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/ServiceCart")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    cart_contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/ServiceCart/cart_base", history_length=3)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


@configclass
class EventCfg:
    """Low-randomization reset events for the first push-cart skeleton."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-0.2, 0.2),
        },
    )
    reset_cart = EventTerm(
        func=mdp.reset_cart_slider,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("cart", joint_names=["slider_joint"])},
    )


@configclass
class CommandsCfg:
    """Approach and push command for the locomotion backbone."""

    # 第一版 push-cart 仍然使用 locomotion 的速度指令。
    # 注意：这还不是 cart-aware command。策略只是被要求向前走，推车交互
    # 主要由下面的 reward 引导。如果 play 不稳定，先确认机器人能否在这个
    # 速度范围内稳定存活，再继续增大推车相关 reward。
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.15, 0.45), lin_vel_y=(-0.05, 0.05), ang_vel_z=(-0.05, 0.05)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.6), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
    )


@configclass
class ActionsCfg:
    """G1 joint-position actions, unchanged from the locomotion baseline."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-1.0, 1.0)},
    )


@configclass
class ObservationsCfg:
    """Observation groups for the push-cart skeleton."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        cart_relative_position = ObsTerm(func=mdp.cart_relative_position_b, clip=(-4.0, 4.0), scale=0.5)
        cart_progress = ObsTerm(func=mdp.cart_joint_position, clip=(-0.5, 3.0), scale=0.5)
        cart_velocity = ObsTerm(func=mdp.cart_joint_velocity, clip=(-3.0, 3.0), scale=0.5)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        cart_relative_position = ObsTerm(func=mdp.cart_relative_position_b, clip=(-4.0, 4.0), scale=0.5)
        cart_position = ObsTerm(func=mdp.cart_position_w, clip=(-5.0, 5.0), scale=0.2)
        cart_velocity = ObsTerm(func=mdp.cart_velocity_w, clip=(-3.0, 3.0), scale=0.5)
        cart_progress = ObsTerm(func=mdp.cart_joint_position, clip=(-0.5, 3.0), scale=0.5)
        cart_slider_velocity = ObsTerm(func=mdp.cart_joint_velocity, clip=(-3.0, 3.0), scale=0.5)
        robot_cart_distance = ObsTerm(func=mdp.robot_cart_distance, clip=(0.0, 4.0), scale=0.5)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Small, interpretable reward set for push-cart v1.0."""

    # 当前诊断重点：
    # - 推车任务不应该直接依赖一个过弱的 locomotion reward 从零学会站稳。
    # - 如果 play 里机器人直接倒，优先看稳定项：
    #   Train/mean_episode_length、Episode_Termination/bad_orientation、
    #   Episode_Reward/flat_orientation_l2、Episode_Reward/base_height、
    #   Episode_Reward/action_rate。
    # - 如果机器人能存活但不会推车，再看任务项：
    #   Episode_Reward/cart_forward_progress 和
    #   Episode_Reward/robot_cart_distance。如果它们长期接近 0，说明推车目标
    #   没有形成有效学习信号。
    # - 这里先把 G1 velocity baseline 中关键的步态、脚部滑移、脚部抬高、
    #   非期望接触、关节偏离项加回来，再逐步调推车项。

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # locomotion 主任务：先保留 baseline 的速度跟踪强度，避免策略只学到
    # “活着但不会稳定走”的局部解。
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # 推车任务项：第一阶段只鼓励靠近和沿 +X 推动，不加入复杂接触分布奖励。
    # 训练时要单独观察这两个标量；总 reward 上升并不代表推车已经学会。
    cart_forward_progress = RewTerm(
        func=mdp.cart_forward_velocity,
        weight=0.8,
        params={"asset_cfg": SceneEntityCfg("cart", joint_names=["slider_joint"]), "max_reward": 0.8},
    )
    robot_cart_distance = RewTerm(
        func=mdp.robot_cart_distance_shaping,
        weight=0.5,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "cart_cfg": SceneEntityCfg("cart", body_names="cart_base"),
            "target_distance": 0.75,
            "max_distance": 2.5,
        },
    )

    # base 稳定性和动作平滑项：基本对齐 G1 velocity baseline。
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2_clipped, weight=-0.01, params={"max_value": 100.0})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2.0e-5)

    # 上半身和腰部先不要自由乱摆。后续如果加入 arm residual action，再单独
    # 放松对应关节的偏离惩罚。
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # 身体姿态和高度：推车前先要求躯干保持接近直立和合理高度。
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.78})

    # 足部 shaping：这些项对 G1 从零学出可用步态很关键。当前推车任务先复用
    # baseline 设置，避免把“不会走路”误判成“不会推车”。
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # 避免躯干、腿部非脚端等部位把地面接触当成“捷径”。
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class PushCartEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the first G1 push-cart loco-manipulation task."""

    scene: PushCartSceneCfg = PushCartSceneCfg(num_envs=2048, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.cart_contact_forces.update_period = self.sim.dt
        # debug 默认关闭，通过 scripts/train.py 显式打开：
        # UNITREE_RSL_RL_DEBUG_TENSORS=1 会检查 obs/reward/action 是否有限。
        # TODO: 需要时增加 cart pose/contact 的任务内统计。
        # TODO: 后续上半身操作可拆成 arm residual action。


@configclass
class PushCartPlayEnvCfg(PushCartEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # play 使用更宽的命令范围，便于手动观察策略边界。调试弱 checkpoint 时，
        # 可以临时改回训练范围，再判断策略是不是真的不会站。
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
