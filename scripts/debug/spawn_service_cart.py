#!/usr/bin/env python3
"""Spawn and inspect the service cart asset in a minimal Isaac Lab scene.

This script is independent from training tasks. It verifies that the
in-repository URDF can be imported, spawned as an articulation, reset, stepped,
and queried for finite state.

Examples:

    python scripts/debug/spawn_service_cart.py --num_envs 1
    python scripts/debug/spawn_service_cart.py --headless --seconds 3
    python scripts/debug/spawn_service_cart.py --headless --animate_slider
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn and inspect the service cart asset.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of scene environments.")
parser.add_argument("--env_spacing", type=float, default=2.5, help="Spacing between environments.")
parser.add_argument("--seconds", type=float, default=5.0, help="Simulation duration.")
parser.add_argument("--animate_slider", action="store_true", help="Directly animate slider_joint with a sine wave.")
parser.add_argument("--slider_amplitude", type=float, default=0.5, help="Slider animation amplitude in meters.")
parser.add_argument("--slider_period", type=float, default=4.0, help="Slider animation period in seconds.")
parser.add_argument("--print_every", type=float, default=1.0, help="Status print interval in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from g1_loco_manipulation.assets.objects.service_cart.service_cart import (
    SERVICE_CART_ASSET_DIR,
    SERVICE_CART_V1_CFG,
)


@configclass
class ServiceCartDebugSceneCfg(InteractiveSceneCfg):
    """Minimal scene containing ground, light, and the service cart."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))
    cart: ArticulationCfg = SERVICE_CART_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/ServiceCart")


def _assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.all(torch.isfinite(value)):
        bad_count = int((~torch.isfinite(value)).sum().item())
        raise RuntimeError(f"{name} contains {bad_count} non-finite values.")


def _print_cart_metadata(cart: Articulation) -> None:
    print("[INFO] Service cart spawned.")
    print(f"[INFO] URDF: {SERVICE_CART_ASSET_DIR / 'service_cart_v1.urdf'}")
    print(f"[INFO] body_names: {cart.body_names}")
    print(f"[INFO] joint_names: {cart.joint_names}")
    print(f"[INFO] default_joint_pos: {cart.data.default_joint_pos.detach().cpu().tolist()}")
    print(f"[INFO] default_joint_vel: {cart.data.default_joint_vel.detach().cpu().tolist()}")
    if "cart_base" not in cart.body_names:
        raise RuntimeError("Expected body 'cart_base' was not found.")
    if "slider_joint" not in cart.joint_names:
        raise RuntimeError("Expected joint 'slider_joint' was not found.")
    _assert_finite("default_joint_pos", cart.data.default_joint_pos)
    _assert_finite("default_joint_vel", cart.data.default_joint_vel)


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """Run a short cart asset smoke test."""

    cart: Articulation = scene["cart"]
    _print_cart_metadata(cart)

    sim_dt = sim.get_physics_dt()
    max_steps = max(1, int(args_cli.seconds / sim_dt))
    print_interval = max(1, int(args_cli.print_every / sim_dt))
    joint_pos = cart.data.default_joint_pos.clone()
    joint_vel = cart.data.default_joint_vel.clone()

    for step_count in range(max_steps):
        if not simulation_app.is_running():
            break

        if args_cli.animate_slider and cart.num_joints > 0:
            t = step_count * sim_dt
            phase = 2.0 * math.pi * t / max(args_cli.slider_period, 1.0e-6)
            joint_pos[:, 0] = args_cli.slider_amplitude * math.sin(phase)
            joint_vel[:, 0] = (
                args_cli.slider_amplitude * 2.0 * math.pi / max(args_cli.slider_period, 1.0e-6) * math.cos(phase)
            )
            cart.write_joint_state_to_sim(joint_pos, joint_vel)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        _assert_finite("cart.root_pos_w", cart.data.root_pos_w)
        _assert_finite("cart.joint_pos", cart.data.joint_pos)
        _assert_finite("cart.joint_vel", cart.data.joint_vel)

        if step_count % print_interval == 0:
            slider_pos = cart.data.joint_pos.detach().cpu().tolist()
            root_pos = cart.data.root_pos_w.detach().cpu().tolist()
            print(f"[INFO] t={step_count * sim_dt:.2f}s joint_pos={slider_pos} root_pos_w={root_pos}")

    print("[INFO] Service cart asset debug test completed successfully.")


def main() -> None:
    urdf_path = SERVICE_CART_ASSET_DIR / "service_cart_v1.urdf"
    print(f"[INFO] Checking service cart URDF: {urdf_path}")
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Service cart URDF not found: {urdf_path}")

    print(f"[INFO] Creating SimulationContext on device: {args_cli.device}")
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    print(f"[INFO] Creating InteractiveScene with num_envs={args_cli.num_envs}")
    scene_cfg = ServiceCartDebugSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)
    print("[INFO] Resetting simulation.")
    sim.reset()
    print("[INFO] Simulation reset complete.")
    sim.set_camera_view([2.4, 1.8, 1.2], [0.0, 0.0, 0.25])

    with torch.inference_mode():
        run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
