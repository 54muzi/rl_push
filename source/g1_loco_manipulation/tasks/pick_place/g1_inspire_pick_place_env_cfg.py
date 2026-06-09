"""G1 Inspire fixed-base pick-place task.

这是从 WBC-AGILE pick-place 任务迁移出的第一阶段可运行版本。它不是
WBC-AGILE 原版 tracking-imitation 任务的逐行复刻，而是为了本仓库的干净
公开代码轨道做了依赖收缩：

- WBC-AGILE 原版依赖 motion yaml、TrackingCommand、lower-body teacher policy、
  远程 table/YCB USD，以及自定义 agile MDP/action/reward。
- 当前任务只使用本仓库的 G1 Inspire USD、本地 primitive table/cube、Isaac Lab
  原生 JointPositionAction 和本任务内的少量 observation/reward helper。
- WBC-AGILE 原版是“跟踪参考轨迹 + 物体轨迹”的 imitation 任务；当前任务是
  固定基座、右臂加右 Inspire 手的 object-centric RL skeleton，用来先确认资产、
  多环境克隆、动作维度、观测奖励和 RSL-RL 链路都能跑通。
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from g1_loco_manipulation.assets.robots.g1 import UNITREE_G1_29DOF_WHOLEBODY_INSPIRE_CFG as ROBOT_CFG
from g1_loco_manipulation.tasks.pick_place import mdp


TABLE_HEIGHT = 0.76
OBJECT_INIT_POS = (0.40, 0.18, TABLE_HEIGHT + 0.045)
PLACE_TARGET_POS = (0.08, 0.18, TABLE_HEIGHT + 0.015)

# 当前动作只控制右臂和右 Inspire 手。WBC-AGILE 原版还有 lower-body policy
# 维持全身运动/站立，这个仓库暂时不引入外部 teacher policy，所以先固定基座。
RIGHT_ARM_HAND_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "R_.*_joint",
]

POSTURE_JOINT_NAMES = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
    "waist_.*_joint",
    "left_shoulder_.*_joint",
    "left_elbow_joint",
    "left_wrist_.*_joint",
    "L_.*_joint",
]

@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Flat scene with G1 Inspire, a table, an object, and a place marker."""

    # 不使用 TerrainImporter 的 generator flat terrain：
    # - generator 的 terrain origins 在单块平地时会全是 0，reset 时会让多 env
    #   的 robot/object 写回同一个位置。
    # - terrain_type="plane" 在 Isaac Sim 5.1 会尝试加载远程默认 ground USD。
    # 所以这里用一个本地 kinematic cuboid 当全局地板，让 env origins 来自 GridCloner。
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # WBC-AGILE 使用远程 table USD；当前任务使用本地 cuboid，避免把外部资产仓库
    # 或 Nucleus/S3 依赖带进这个公开代码轨道。
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 0.55, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.38, 0.36, 0.32)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, 0.18, TABLE_HEIGHT - 0.04)),
    )
    # WBC-AGILE 随机 YCB/DexCube USD；当前先用单个 primitive cube，便于排查
    # contact、reset 高度和 reward 是否合理。后续可在这里替换为本地对象资产集。
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.4, dynamic_friction=1.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.52, 0.80)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_INIT_POS, rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # 这里只是可视化放置目标，不参与碰撞。真正的任务信号来自 observation/reward
    # 里的 object_to_target 和 object_target_distance。
    target_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PlaceTarget",
        spawn=sim_utils.CuboidCfg(
            size=(0.14, 0.14, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.75, 0.30)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=PLACE_TARGET_POS),
    )
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
    """No sampled command is used in the first fixed-base pick-place stage."""

    # WBC-AGILE 的 CommandsCfg 核心是 TrackingCommand：从 motion yaml 读取
    # 机器人和物体参考轨迹，并在 obs/reward/reset 中反复使用。
    # 当前任务没有引入 motion 文件，所以这里保持为空。


@configclass
class ActionsCfg:
    """Right arm and right Inspire hand joint-position action."""

    # WBC-AGILE 原版上半身使用 delta joint action，下半身使用加载 checkpoint 的
    # AgileLowerBodyAction。当前任务为了自包含，直接用 JointPositionAction 控制
    # 右臂和右手，scale 表示相对默认姿态的目标范围。
    right_arm_hand = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_HAND_JOINT_NAMES,
        scale=0.35,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-1.0, 1.0)},
    )


@configclass
class ObservationsCfg:
    """Policy and critic observations for object-centric pick-place."""

    @configclass
    class PolicyCfg(ObsGroup):
        # 当前 policy obs 保留 proprioception + object/wrist/target 几何关系。
        # WBC-AGILE 原版额外包含 motion anchor、trajectory delta、lower-body obs
        # 等 imitation 信息；这里不加入这些项，避免观察语义伪装成原版 tracking。
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.0, n_max=1.0))
        object_position = ObsTerm(func=mdp.object_position_b, clip=(-3.0, 3.0), scale=1.0)
        object_to_wrist = ObsTerm(
            func=mdp.object_to_wrist_b,
            params={"robot_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link")},
            clip=(-3.0, 3.0),
            scale=1.0,
        )
        target_position = ObsTerm(
            func=mdp.target_position_b,
            params={"target_pos": PLACE_TARGET_POS},
            clip=(-3.0, 3.0),
            scale=1.0,
        )
        object_to_target = ObsTerm(
            func=mdp.object_to_target,
            params={"target_pos": PLACE_TARGET_POS},
            clip=(-3.0, 3.0),
            scale=1.0,
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        # critic 多看一些诊断量，例如 wrist_position/object_height，但仍不引入
        # WBC-AGILE motion command 的 privileged trajectory state。
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        wrist_position = ObsTerm(
            func=mdp.wrist_position_b,
            params={"robot_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link")},
            clip=(-3.0, 3.0),
        )
        object_position = ObsTerm(func=mdp.object_position_b, clip=(-3.0, 3.0))
        object_height = ObsTerm(func=mdp.object_height, clip=(0.0, 3.0))
        target_position = ObsTerm(func=mdp.target_position_b, params={"target_pos": PLACE_TARGET_POS}, clip=(-3.0, 3.0))
        object_to_target = ObsTerm(func=mdp.object_to_target, params={"target_pos": PLACE_TARGET_POS}, clip=(-3.0, 3.0))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Sparse enough to represent pick-place, shaped enough for first smoke training."""

    # WBC-AGILE 的主要 reward 是“手/关节/根节点/物体轨迹跟踪 + 稳定正则”。
    # 当前任务没有参考轨迹，所以奖励只表达三个阶段：
    # 1. 右腕接近物体；2. 物体高于桌面；3. 物体靠近放置目标。
    # 这些 shaping 用于跑通训练链路，不代表已经具备完整灵巧抓取能力。
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    hand_object_distance = RewTerm(
        func=mdp.hand_object_distance_exp,
        weight=1.5,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            "std": 0.25,
        },
    )
    object_lifted = RewTerm(
        func=mdp.object_lifted,
        weight=2.0,
        params={"table_height": TABLE_HEIGHT, "lift_height": 0.12},
    )
    object_target_distance = RewTerm(
        func=mdp.object_target_distance_exp,
        weight=3.0,
        params={"target_pos": PLACE_TARGET_POS, "std": 0.32},
    )
    object_not_dropped = RewTerm(func=mdp.object_not_dropped, weight=0.2, params={"minimum_height": TABLE_HEIGHT - 0.08})
    action_rate = RewTerm(func=mdp.action_rate_l2_clipped, weight=-0.02, params={"max_value": 100.0})
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_HAND_JOINT_NAMES)},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_HAND_JOINT_NAMES)},
    )
    joint_deviation_posture = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=POSTURE_JOINT_NAMES)},
    )


@configclass
class TerminationsCfg:
    """Fixed-base first stage only times out."""

    # 暂时只保留 timeout。若后续解除固定基座或加入更复杂物体随机化，再加
    # base 姿态、物体掉落、越界等 termination。
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """Low-randomization resets for the initial migration."""

    # 注意 reset_root_state_uniform 的 pose_range 是相对 init_state 的偏移，
    # 不是绝对世界坐标。这里写 0 偏移，避免把 init_state 加两次。
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "yaw": (0.0, 0.0),
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
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
        },
    )


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 Inspire fixed-base pick-place task."""

    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.ground.spawn.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 本仓库 Inspire USD cfg 默认带 articulation_root_prim_path="/pelvis"。
        # 多 env 下这里让 Isaac Lab 按 "{ENV_REGEX_NS}/Robot" 自动解析 articulation root，
        # 避免绝对 root path 破坏克隆环境中的 articulation 绑定。
        self.scene.robot.articulation_root_prim_path = None
        self.scene.robot.init_state.pos = (0.15, -0.50, 0.80)
        self.scene.robot.init_state.rot = (0.7071068, 0.0, 0.0, 0.7071068)
        # 先固定基座：这样可以单独验证 Inspire 手模型、右臂动作、物体接触和
        # observation/reward。WBC-AGILE 的全身版本需要 lower-body teacher policy，
        # 当前仓库按规则不提交 checkpoint，因此不在这里隐式依赖它。
        self.scene.robot.spawn.fix_base = True
        self.scene.robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        )
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class PickPlacePlayEnvCfg(PickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # play 使用较少 env 便于 GUI 观察；观测/动作维度保持和训练 cfg 一致，
        # 否则 checkpoint 加载会出现模型输入/输出维度不匹配。
        self.scene.num_envs = 16
