#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd_path", type=str, required=True)
parser.add_argument("--body_filter", type=str, default=".*", help="Regex filter for body names.")
parser.add_argument("--marker_radius", type=float, default=0.025)
parser.add_argument("--seconds", type=float, default=60.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationCfg, SimulationContext


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def create_sphere_marker(stage, prim_path: str, pos, radius: float):
    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.CreateRadiusAttr(radius)
    UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    return sphere


def main():
    usd_path = str(Path(args_cli.usd_path).expanduser().resolve())
    if not Path(usd_path).is_file():
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    sim = SimulationContext(SimulationCfg(dt=0.005, device=args_cli.device))
    sim.set_camera_view([2.5, 2.5, 1.5], [0.0, 0.0, 0.7])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=1000.0)
    light_cfg.func("/World/skyLight", light_cfg)

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={
                # 第一版先让手部关节归零，避免奇怪默认姿态。
                "L_.*_joint": 0.0,
                "R_.*_joint": 0.0,
            },
        ),
        actuators={
            # 这里只用于可视化和 smoke test，不是最终训练参数。
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=300.0,
                velocity_limit_sim=100.0,
                stiffness=30.0,
                damping=3.0,
            ),
        },
    )

    robot = Articulation(robot_cfg)

    sim.reset()
    robot.update(sim.get_physics_dt())

    body_regex = re.compile(args_cli.body_filter)

    print("\n========== BODY MARKERS ==========")
    print("index | body_name | marker_prim")
    print("----------------------------------")

    stage = sim.stage
    marker_root = "/World/DebugBodyMarkers"
    sim_utils.create_prim(marker_root, "Xform")

    body_pos_w = robot.data.body_pos_w[0].detach().cpu()

    for idx, body_name in enumerate(robot.body_names):
        if not body_regex.match(body_name):
            continue

        pos = body_pos_w[idx].numpy()
        marker_name = f"B{idx:03d}_{sanitize(body_name)}"
        marker_prim = f"{marker_root}/{marker_name}"
        create_sphere_marker(stage, marker_prim, pos, args_cli.marker_radius)

        print(f"{idx:03d} | {body_name} | {marker_prim}")

    print("\n[INFO] Body markers created.")
    print("[INFO] In Isaac Sim viewport, inspect /World/DebugBodyMarkers.")
    print("[INFO] Use the printed index/body mapping to identify the selected location.")

    # GUI 模式下保持运行，方便查看；headless 下跑一段时间后退出。
    max_steps = max(1, int(args_cli.seconds / sim.get_physics_dt()))
    for _ in range(max_steps):
        if not simulation_app.is_running():
            break
        sim.step()
        robot.update(sim.get_physics_dt())

    simulation_app.close()


if __name__ == "__main__":
    main()