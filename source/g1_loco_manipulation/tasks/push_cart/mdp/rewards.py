from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .observations import _first_body_pos, _single_joint_pos, _single_joint_vel

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cart_forward_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cart", joint_names=["slider_joint"]),
    max_reward: float = 1.0,
) -> torch.Tensor:
    """Reward positive cart motion along the rail-constrained +X direction."""
    cart: Articulation = env.scene[asset_cfg.name]
    velocity = _single_joint_vel(cart, asset_cfg).squeeze(-1)
    velocity = torch.nan_to_num(velocity, nan=0.0, posinf=max_reward, neginf=-max_reward)
    # 只有机器人真的推动 slider joint 后，这一项才会明显大于 0。
    # 如果 TensorBoard 中该项长期接近 0，说明策略大概率只学到了存活/速度跟踪，
    # 还没有获得有效的推车学习信号。
    return torch.clamp(velocity, min=0.0, max=max_reward)


def cart_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cart", joint_names=["slider_joint"]),
    target: float = 2.0,
) -> torch.Tensor:
    """Bounded absolute cart progress for privileged diagnostics."""
    cart: Articulation = env.scene[asset_cfg.name]
    progress = _single_joint_pos(cart, asset_cfg).squeeze(-1)
    progress = torch.nan_to_num(progress, nan=0.0, posinf=target, neginf=0.0)
    return torch.clamp(progress / max(target, 1.0e-6), min=0.0, max=1.0)


def robot_cart_distance_shaping(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cart_cfg: SceneEntityCfg = SceneEntityCfg("cart", body_names="cart_base"),
    target_distance: float = 0.55,
    max_distance: float = 2.5,
) -> torch.Tensor:
    """Encourage the robot to approach the cart without requiring hard contact."""
    robot: Articulation = env.scene[robot_cfg.name]
    cart: Articulation = env.scene[cart_cfg.name]
    rel_xy = _first_body_pos(cart, cart_cfg)[:, :2] - robot.data.root_pos_w[:, :2]
    distance = torch.linalg.norm(rel_xy, dim=-1)
    distance = torch.nan_to_num(distance, nan=4.0, posinf=4.0, neginf=0.0)
    # 第一版不要用太尖锐的高斯距离奖励。距离较远时高斯项会接近 0，策略很难
    # 分辨“走近一点”是否有收益。这里使用线性 shaping，保证接近 cart 的过程
    # 有连续梯度；进入 target_distance 内后给满分，但不强迫持续硬接触。
    span = max(max_distance - target_distance, 1.0e-6)
    error = torch.clamp(distance - target_distance, min=0.0, max=span)
    return 1.0 - error / span
