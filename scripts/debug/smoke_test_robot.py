"""Smoke test for Unitree G1 robot assets.
Examples:
    python scripts/debug/smoke_test_robot.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test Unitree G1 robot assets.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from g1_loco_manipulation.assets.robots import UNITREE_G1_29DOF_WHOLEBODY_INSPIRE_CFG, UNITREE_G1_29DOF_CFG

sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([3.0, 2.5, 1.6], [0.0, 0.0, 0.7])

# 最小可视化场景：用本地几何体做地面，避免 GroundPlaneCfg 依赖 Isaac Nucleus USD 路径。
ground_cfg = sim_utils.CuboidCfg(
    size=(20.0, 20.0, 0.02),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
)
ground_cfg.func("/World/defaultGroundPlane", ground_cfg, translation=(0.0, 0.0, -0.01))
light_cfg = sim_utils.DomeLightCfg(intensity=1000.0)
light_cfg.func("/World/skyLight", light_cfg)

robot_cfg = UNITREE_G1_29DOF_WHOLEBODY_INSPIRE_CFG.replace(prim_path="/World/Robot")

robot = Articulation(robot_cfg)
sim.reset()


# 打印关节名称，验证是否正确加载了 USD 中的关节信息。
print("\nJOINT NAMES:")
for i, name in enumerate(robot.joint_names):
    print(i, name)

# 打印身体名称，验证是否正确加载了 USD 中的身体信息。
print("\nBODY NAMES:")
for i, name in enumerate(robot.body_names):
    print(i, name)

# 打印关节软限制，验证是否正确解析了 USD 中的限制信息。
print("\nsoft joint pos limits:")
for i, name in enumerate(robot.joint_names):
    print(i, name, robot.data.soft_joint_pos_limits[0, i].cpu().numpy())

# 打印默认关节位置，验证是否正确应用了 USD 中的初始状态设置。
print("\ndefault joint pos:")
for i, name in enumerate(robot.joint_names):
    print(i, name, float(robot.data.default_joint_pos[0, i]))

# 打印重置后关节位置，验证是否正确应用了 USD 中的重置状态设置。
print("\nDEFAULT JOINT POS AFTER RESET:")
for i, name in enumerate(robot.joint_names):
    print(i, name, float(robot.data.joint_pos[0, i]))


for i in range(100):
    sim.step()
    robot.update(sim.get_physics_dt())

    q = robot.data.joint_pos
    qd = robot.data.joint_vel
    root = robot.data.root_state_w

    if torch.isnan(q).any() or torch.isnan(qd).any() or torch.isnan(root).any():
        raise RuntimeError(f"NaN detected at step {i}")

    if torch.isinf(q).any() or torch.isinf(qd).any() or torch.isinf(root).any():
        raise RuntimeError(f"Inf detected at step {i}")

print("[OK] Robot USD loaded and simulated for 100 steps without NaN/Inf.")
simulation_app.close()
