from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .observations import _body_pos, _target_pos_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hand_object_distance_exp(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 0.25,
) -> torch.Tensor:
    """Reward the right wrist approaching the object."""
    robot = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    distance = torch.linalg.norm(_body_pos(robot, robot_cfg) - obj.data.root_pos_w, dim=-1)
    distance = torch.nan_to_num(distance, nan=4.0, posinf=4.0, neginf=0.0)
    return torch.exp(-(distance**2) / max(std**2, 1.0e-6))


def object_lifted(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    table_height: float = 0.76,
    lift_height: float = 0.10,
) -> torch.Tensor:
    """Bounded reward for lifting the object above the table top."""
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2] - table_height
    height = torch.nan_to_num(height, nan=0.0, posinf=lift_height, neginf=0.0)
    return torch.clamp(height / max(lift_height, 1.0e-6), min=0.0, max=1.0)


def object_target_distance_exp(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 0.30,
) -> torch.Tensor:
    """Reward the object being near the place target."""
    obj: RigidObject = env.scene[object_cfg.name]
    distance = torch.linalg.norm(obj.data.root_pos_w - _target_pos_w(env, target_pos), dim=-1)
    distance = torch.nan_to_num(distance, nan=4.0, posinf=4.0, neginf=0.0)
    return torch.exp(-(distance**2) / max(std**2, 1.0e-6))


def object_not_dropped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    minimum_height: float = 0.70,
) -> torch.Tensor:
    """Soft survival term for keeping the object on or above the table."""
    obj: RigidObject = env.scene[object_cfg.name]
    height = torch.nan_to_num(obj.data.root_pos_w[:, 2], nan=0.0, posinf=minimum_height, neginf=0.0)
    return (height > minimum_height).float()
