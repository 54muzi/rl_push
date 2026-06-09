"""Configuration for the push-cart v1.0 asset.

This asset is intentionally a rail-constrained pushable cart, not a realistic
four-wheel dynamics model. The rectangular cart body and four fixed wheel
cylinders are imported from URDF as a single child link of a passive prismatic
joint. The joint constrains motion to the cart's local X axis so the first
push-cart task can focus on stable robot-object interaction.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

SERVICE_CART_ASSET_DIR = Path(__file__).resolve().parent / "urdf"


@configclass
class ServiceCartUrdfFileCfg(sim_utils.UrdfFileCfg):
    """URDF spawn settings for the constrained service cart."""

    fix_base: bool = True
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = False
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        target_type="none",
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.08,
        angular_damping=0.4,
        max_linear_velocity=3.0,
        max_angular_velocity=8.0,
        max_depenetration_velocity=1.0,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )


SERVICE_CART_V1_CFG = ArticulationCfg(
    spawn=ServiceCartUrdfFileCfg(
        asset_path=str(SERVICE_CART_ASSET_DIR / "service_cart_v1.urdf"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={"slider_joint": 0.0},
        joint_vel={"slider_joint": 0.0},
    ),
    actuators={
        "passive_slider": ImplicitActuatorCfg(
            joint_names_expr=["slider_joint"],
            effort_limit_sim=0.0,
            velocity_limit_sim=3.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
