# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree G1 robots used by this project."""

from __future__ import annotations

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

G1_ROBOT_ASSET_DIR = Path(__file__).resolve().parent
UNITREE_ROS_DIR = Path(os.environ.get("UNITREE_ROS_DIR", str(G1_ROBOT_ASSET_DIR / "unitree_ros")))
UNITREE_G1_INSPIRE_DIR = Path(
    os.environ.get("UNITREE_G1_INSPIRE_DIR", str(G1_ROBOT_ASSET_DIR / "g1-29dof_wholebody_inspire"))
)
UNITREE_G1_INSPIRE_USD_PATH = os.environ.get(
    "UNITREE_G1_INSPIRE_USD_PATH",
    str(UNITREE_G1_INSPIRE_DIR / "g1_29dof_with_inspire_rev_1_0_no_hand_camera.usd"),
)

# 关节分区只描述身体部位，不绑定具体 actuator 参数，方便任务配置复用。
LEG_JOINT_NAMES = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
]
LEG_MOTOR_JOINT_NAMES = [
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
]
ANKLE_JOINT_NAMES = [
    ".*_ankle_.*_joint",
]
FEET_LINK_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]
WAIST_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]
ARM_JOINT_NAMES = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
]
HAND_JOINT_NAMES = [
    "L_.*_joint",
    "R_.*_joint",
]
NO_HAND_JOINT_NAMES = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
    "waist_.*_joint",
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
]
RIGHT_HAND_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "R_.*_joint",
]
LEFT_HAND_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "L_.*_joint",
]

# stiffness/damping 不直接手写，而是由电机等效转动惯量、目标自然频率和阻尼比派生：
#   stiffness = armature * natural_frequency^2
#   damping   = 2 * damping_ratio * armature * natural_frequency
# 这样保留每类电机的物理含义，后续只需调频率或阻尼比即可整体调整动态响应。
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
ARMATURE_1515 = 0.00149

NATURAL_FREQ = 10.0 * 2.0 * math.pi  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2
STIFFNESS_1515 = 2.0

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ
DAMPING_1515 = 0.2
EFFORT_LIMIT_1515 = 0.76
VELOCITY_LIMIT_1515 = 23.0

G1_LEG_EFFORT_LIMIT_SIM = {
    ".*_hip_yaw_joint": 88.0,
    ".*_hip_roll_joint": 139.0,
    ".*_hip_pitch_joint": 88.0,
    ".*_knee_joint": 139.0,
}
G1_LEG_VELOCITY_LIMIT_SIM = {
    ".*_hip_yaw_joint": 32.0,
    ".*_hip_roll_joint": 20.0,
    ".*_hip_pitch_joint": 32.0,
    ".*_knee_joint": 20.0,
}
G1_LEG_STIFFNESS = {
    ".*_hip_yaw_joint": STIFFNESS_7520_14,
    ".*_hip_roll_joint": STIFFNESS_7520_22,
    ".*_hip_pitch_joint": STIFFNESS_7520_14,
    ".*_knee_joint": STIFFNESS_7520_22,
}
G1_LEG_DAMPING = {
    ".*_hip_yaw_joint": DAMPING_7520_14,
    ".*_hip_roll_joint": DAMPING_7520_22,
    ".*_hip_pitch_joint": DAMPING_7520_14,
    ".*_knee_joint": DAMPING_7520_22,
}
G1_LEG_ARMATURE = {
    ".*_hip_yaw_joint": ARMATURE_7520_14,
    ".*_hip_roll_joint": ARMATURE_7520_22,
    ".*_hip_pitch_joint": ARMATURE_7520_14,
    ".*_knee_joint": ARMATURE_7520_22,
}

G1_ANKLE_EFFORT_LIMIT_SIM = 50.0
G1_ANKLE_VELOCITY_LIMIT_SIM = 37.0
G1_ANKLE_STIFFNESS = 2.0 * STIFFNESS_5020
G1_ANKLE_DAMPING = 2.0 * DAMPING_5020
G1_ANKLE_ARMATURE = 2.0 * ARMATURE_5020

G1_WAIST_EFFORT_LIMIT_SIM = {
    "waist_yaw_joint": 88.0,
    "waist_roll_joint": 50.0,
    "waist_pitch_joint": 50.0,
}
G1_WAIST_VELOCITY_LIMIT_SIM = {
    "waist_yaw_joint": 32.0,
    "waist_roll_joint": 37.0,
    "waist_pitch_joint": 37.0,
}
G1_WAIST_STIFFNESS = {
    "waist_yaw_joint": STIFFNESS_7520_14,
    "waist_roll_joint": 2.0 * STIFFNESS_5020,
    "waist_pitch_joint": 2.0 * STIFFNESS_5020,
}
G1_WAIST_DAMPING = {
    "waist_yaw_joint": DAMPING_7520_14,
    "waist_roll_joint": 2.0 * DAMPING_5020,
    "waist_pitch_joint": 2.0 * DAMPING_5020,
}
G1_WAIST_ARMATURE = {
    "waist_yaw_joint": ARMATURE_7520_14,
    "waist_roll_joint": 2.0 * ARMATURE_5020,
    "waist_pitch_joint": 2.0 * ARMATURE_5020,
}

G1_ARM_EFFORT_LIMIT_SIM = {
    ".*_shoulder_pitch_joint": 25.0,
    ".*_shoulder_roll_joint": 25.0,
    ".*_shoulder_yaw_joint": 25.0,
    ".*_elbow_joint": 25.0,
    ".*_wrist_roll_joint": 25.0,
    ".*_wrist_pitch_joint": 5.0,
    ".*_wrist_yaw_joint": 5.0,
}
G1_ARM_VELOCITY_LIMIT_SIM = {
    ".*_shoulder_pitch_joint": 37.0,
    ".*_shoulder_roll_joint": 37.0,
    ".*_shoulder_yaw_joint": 37.0,
    ".*_elbow_joint": 37.0,
    ".*_wrist_roll_joint": 37.0,
    ".*_wrist_pitch_joint": 22.0,
    ".*_wrist_yaw_joint": 22.0,
}
G1_ARM_STIFFNESS = {
    ".*_shoulder_pitch_joint": STIFFNESS_5020,
    ".*_shoulder_roll_joint": STIFFNESS_5020,
    ".*_shoulder_yaw_joint": STIFFNESS_5020,
    ".*_elbow_joint": STIFFNESS_5020,
    ".*_wrist_roll_joint": STIFFNESS_5020,
    ".*_wrist_pitch_joint": STIFFNESS_4010,
    ".*_wrist_yaw_joint": STIFFNESS_4010,
}
G1_ARM_DAMPING = {
    ".*_shoulder_pitch_joint": DAMPING_5020,
    ".*_shoulder_roll_joint": DAMPING_5020,
    ".*_shoulder_yaw_joint": DAMPING_5020,
    ".*_elbow_joint": DAMPING_5020,
    ".*_wrist_roll_joint": DAMPING_5020,
    ".*_wrist_pitch_joint": DAMPING_4010,
    ".*_wrist_yaw_joint": DAMPING_4010,
}
G1_ARM_ARMATURE = {
    ".*_shoulder_pitch_joint": ARMATURE_5020,
    ".*_shoulder_roll_joint": ARMATURE_5020,
    ".*_shoulder_yaw_joint": ARMATURE_5020,
    ".*_elbow_joint": ARMATURE_5020,
    ".*_wrist_roll_joint": ARMATURE_5020,
    ".*_wrist_pitch_joint": ARMATURE_4010,
    ".*_wrist_yaw_joint": ARMATURE_4010,
}

G1_HAND_EFFORT_LIMIT_SIM = EFFORT_LIMIT_1515
G1_HAND_VELOCITY_LIMIT_SIM = VELOCITY_LIMIT_1515
G1_HAND_STIFFNESS = STIFFNESS_1515
G1_HAND_DAMPING = DAMPING_1515
G1_HAND_ARMATURE = ARMATURE_1515


@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None
    soft_joint_pos_limit_factor = 0.9


@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    """USD spawn settings shared by Unitree robot assets."""

    activate_contact_sensors: bool = True
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )


@configclass
class UnitreeUrdfFileCfg(sim_utils.UrdfFileCfg):
    """URDF spawn settings for Unitree robot descriptions."""

    fix_base: bool = False
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = True
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )


UNITREE_G1_29DOF_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=str(UNITREE_ROS_DIR / "robots" / "g1_description" / "g1_29dof_rev_1_0.urdf"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=LEG_MOTOR_JOINT_NAMES,
            effort_limit_sim=G1_LEG_EFFORT_LIMIT_SIM,
            velocity_limit_sim=G1_LEG_VELOCITY_LIMIT_SIM,
            stiffness=G1_LEG_STIFFNESS,
            damping=G1_LEG_DAMPING,
            armature=G1_LEG_ARMATURE,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=ANKLE_JOINT_NAMES,
            effort_limit_sim=G1_ANKLE_EFFORT_LIMIT_SIM,
            velocity_limit_sim=G1_ANKLE_VELOCITY_LIMIT_SIM,
            stiffness=G1_ANKLE_STIFFNESS,
            damping=G1_ANKLE_DAMPING,
            armature=G1_ANKLE_ARMATURE,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=WAIST_JOINT_NAMES,
            effort_limit_sim=G1_WAIST_EFFORT_LIMIT_SIM,
            velocity_limit_sim=G1_WAIST_VELOCITY_LIMIT_SIM,
            stiffness=G1_WAIST_STIFFNESS,
            damping=G1_WAIST_DAMPING,
            armature=G1_WAIST_ARMATURE,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            effort_limit_sim=G1_ARM_EFFORT_LIMIT_SIM,
            velocity_limit_sim=G1_ARM_VELOCITY_LIMIT_SIM,
            stiffness=G1_ARM_STIFFNESS,
            damping=G1_ARM_DAMPING,
            armature=G1_ARM_ARMATURE,
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
)


UNITREE_G1_29DOF_WHOLEBODY_INSPIRE_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path=str(UNITREE_G1_INSPIRE_USD_PATH),
    ),
    articulation_root_prim_path="/pelvis",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
            "L_.*_joint": 0.0,
            "R_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=LEG_MOTOR_JOINT_NAMES,
            effort_limit_sim=G1_LEG_EFFORT_LIMIT_SIM,
            velocity_limit_sim=G1_LEG_VELOCITY_LIMIT_SIM,
            stiffness=G1_LEG_STIFFNESS,
            damping=G1_LEG_DAMPING,
            armature=G1_LEG_ARMATURE,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=ANKLE_JOINT_NAMES,
            effort_limit_sim=G1_ANKLE_EFFORT_LIMIT_SIM,
            velocity_limit_sim=G1_ANKLE_VELOCITY_LIMIT_SIM,
            stiffness=G1_ANKLE_STIFFNESS,
            damping=G1_ANKLE_DAMPING,
            armature=G1_ANKLE_ARMATURE,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=WAIST_JOINT_NAMES,
            effort_limit_sim=G1_WAIST_EFFORT_LIMIT_SIM,
            velocity_limit_sim=G1_WAIST_VELOCITY_LIMIT_SIM,
            stiffness=G1_WAIST_STIFFNESS,
            damping=G1_WAIST_DAMPING,
            armature=G1_WAIST_ARMATURE,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            effort_limit_sim=G1_ARM_EFFORT_LIMIT_SIM,
            velocity_limit_sim=G1_ARM_VELOCITY_LIMIT_SIM,
            stiffness=G1_ARM_STIFFNESS,
            damping=G1_ARM_DAMPING,
            armature=G1_ARM_ARMATURE,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=HAND_JOINT_NAMES,
            effort_limit_sim=G1_HAND_EFFORT_LIMIT_SIM,
            velocity_limit_sim=G1_HAND_VELOCITY_LIMIT_SIM,
            stiffness=G1_HAND_STIFFNESS,
            damping=G1_HAND_DAMPING,
            armature=G1_HAND_ARMATURE,
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
        "L_index_proximal_joint",
        "L_middle_proximal_joint",
        "L_pinky_proximal_joint",
        "L_ring_proximal_joint",
        "L_thumb_proximal_yaw_joint",
        "R_index_proximal_joint",
        "R_middle_proximal_joint",
        "R_pinky_proximal_joint",
        "R_ring_proximal_joint",
        "R_thumb_proximal_yaw_joint",
        "L_index_intermediate_joint",
        "L_middle_intermediate_joint",
        "L_pinky_intermediate_joint",
        "L_ring_intermediate_joint",
        "L_thumb_proximal_pitch_joint",
        "R_index_intermediate_joint",
        "R_middle_intermediate_joint",
        "R_pinky_intermediate_joint",
        "R_ring_intermediate_joint",
        "R_thumb_proximal_pitch_joint",
        "L_thumb_intermediate_joint",
        "R_thumb_intermediate_joint",
        "L_thumb_distal_joint",
        "R_thumb_distal_joint",
    ],
)


def _action_scale_from_actuators(actuators: dict[str, ImplicitActuatorCfg]) -> dict[str, float]:
    """从 actuator 限幅和刚度推导 action scale。"""

    action_scale = {}
    for actuator_cfg in actuators.values():
        effort = actuator_cfg.effort_limit_sim
        stiffness = actuator_cfg.stiffness
        joint_names = actuator_cfg.joint_names_expr
        if not isinstance(effort, dict):
            effort = dict.fromkeys(joint_names, effort)
        if not isinstance(stiffness, dict):
            stiffness = dict.fromkeys(joint_names, stiffness)
        for joint_name in joint_names:
            joint_effort = effort.get(joint_name)
            joint_stiffness = stiffness.get(joint_name)
            if joint_effort is not None and joint_stiffness:
                action_scale[joint_name] = 0.25 * joint_effort / joint_stiffness
    return action_scale


G1_29DOF_ACTION_SCALE = _action_scale_from_actuators(UNITREE_G1_29DOF_CFG.actuators)
G1_W_HANDS_AGILE_ACTION_SCALE = _action_scale_from_actuators(UNITREE_G1_29DOF_WHOLEBODY_INSPIRE_CFG.actuators)
G1_NO_HANDS_AGILE_ACTION_SCALE = {
    joint_name: scale
    for joint_name, scale in G1_W_HANDS_AGILE_ACTION_SCALE.items()
    if joint_name not in HAND_JOINT_NAMES
}
