from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _finite_clip(value: torch.Tensor, limit: float) -> torch.Tensor:
    value = torch.nan_to_num(value, nan=0.0, posinf=limit, neginf=-limit)
    return torch.clamp(value, -limit, limit)


def _first_body_pos(asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    if asset_cfg.body_ids is None:
        return asset.data.root_pos_w
    if isinstance(asset_cfg.body_ids, slice):
        return asset.data.body_pos_w[:, -1, :]
    return asset.data.body_pos_w[:, asset_cfg.body_ids[0], :]


def _first_body_lin_vel(asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    if asset_cfg.body_ids is None:
        return asset.data.root_lin_vel_w
    if isinstance(asset_cfg.body_ids, slice):
        return asset.data.body_lin_vel_w[:, -1, :]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]


def _single_joint_pos(asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if joint_pos.ndim == 1:
        return joint_pos.unsqueeze(-1)
    return joint_pos[:, :1]


def _single_joint_vel(asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if joint_vel.ndim == 1:
        return joint_vel.unsqueeze(-1)
    return joint_vel[:, :1]


def cart_position_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cart", body_names="cart_base"),
) -> torch.Tensor:
    """Cart body position in world frame, clipped for critic stability."""
    cart: Articulation = env.scene[asset_cfg.name]
    return _finite_clip(_first_body_pos(cart, asset_cfg), 5.0)


def cart_velocity_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cart", body_names="cart_base"),
) -> torch.Tensor:
    """Cart body linear velocity in world frame."""
    cart: Articulation = env.scene[asset_cfg.name]
    return _finite_clip(_first_body_lin_vel(cart, asset_cfg), 3.0)


def cart_relative_position_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cart_cfg: SceneEntityCfg = SceneEntityCfg("cart", body_names="cart_base"),
) -> torch.Tensor:
    """Cart position relative to the robot base, expressed in robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    cart: Articulation = env.scene[cart_cfg.name]
    rel_pos_w = _first_body_pos(cart, cart_cfg) - robot.data.root_pos_w
    rel_pos_b = quat_apply_inverse(robot.data.root_quat_w, rel_pos_w)
    return _finite_clip(rel_pos_b, 4.0)


def cart_joint_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cart", joint_names=["slider_joint"]),
) -> torch.Tensor:
    """Rail slider joint position, used as bounded cart progress."""
    cart: Articulation = env.scene[asset_cfg.name]
    return _finite_clip(_single_joint_pos(cart, asset_cfg), 4.0)


def cart_joint_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cart", joint_names=["slider_joint"]),
) -> torch.Tensor:
    """Rail slider joint velocity."""
    cart: Articulation = env.scene[asset_cfg.name]
    return _finite_clip(_single_joint_vel(cart, asset_cfg), 3.0)


def robot_cart_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cart_cfg: SceneEntityCfg = SceneEntityCfg("cart", body_names="cart_base"),
) -> torch.Tensor:
    """Planar distance between robot base and cart body."""
    robot: Articulation = env.scene[robot_cfg.name]
    cart: Articulation = env.scene[cart_cfg.name]
    rel_xy = _first_body_pos(cart, cart_cfg)[:, :2] - robot.data.root_pos_w[:, :2]
    distance = torch.linalg.norm(rel_xy, dim=-1, keepdim=True)
    return _finite_clip(distance, 4.0)
