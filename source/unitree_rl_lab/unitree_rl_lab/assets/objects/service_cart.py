"""Configuration for a minimal pushable service cart asset.

The v1.0 cart is intentionally simple: one box-shaped body and four fixed
wheel-shaped cylinders. A prismatic joint in the URDF constrains the cart to
move along its longest side, which is the local X axis.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass


SERVICE_CART_ASSET_DIR = os.path.join(os.path.dirname(__file__), "urdf")


@configclass
class ServiceCartUrdfFileCfg(sim_utils.UrdfFileCfg):
    """URDF spawn settings for the constrained service cart."""

    # The URDF has a fixed root link named "world" and a prismatic child joint.
    # Keeping the base fixed leaves only slider_joint free to move.
    fix_base: bool = True
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = False

    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.15,
        angular_damping=0.8,
        max_linear_velocity=5.0,
        max_angular_velocity=10.0,
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
        asset_path=os.path.join(SERVICE_CART_ASSET_DIR, "service_cart_v1.urdf"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={"slider_joint": 0.0},
        joint_vel={"slider_joint": 0.0},
    ),
    actuators={
        # Zero stiffness/effort keeps the X slider passive: robots can push it,
        # but the asset itself does not try to drive the joint.
        "passive_slider": ImplicitActuatorCfg(
            joint_names_expr=["slider_joint"],
            effort_limit=0.0,
            velocity_limit=5.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
