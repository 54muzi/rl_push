from __future__ import annotations

from typing import TYPE_CHECKING

import torch

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _finite_clip(value: torch.Tensor, limit: float) -> torch.Tensor:
    value = torch.nan_to_num(value, nan=0.0, posinf=limit, neginf=-limit)
    return torch.clamp(value, -limit, limit)


def _body_pos(asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    if asset_cfg.body_ids is None:
        return asset.data.root_pos_w
    if isinstance(asset_cfg.body_ids, slice):
        return asset.data.body_pos_w[:, -1, :]
    return asset.data.body_pos_w[:, asset_cfg.body_ids[0], :]


def _target_pos_w(env: ManagerBasedRLEnv, target_pos: tuple[float, float, float]) -> torch.Tensor:
    target = torch.tensor(target_pos, device=env.device, dtype=torch.float32).unsqueeze(0)
    return target + env.scene.env_origins


def object_position_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object position relative to the robot root, expressed in robot-root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    rel_pos_w = obj.data.root_pos_w - robot.data.root_pos_w
    return _finite_clip(quat_apply_inverse(robot.data.root_quat_w, rel_pos_w), 4.0)


def wrist_position_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
) -> torch.Tensor:
    """Right wrist position relative to the robot root, expressed in robot-root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    rel_pos_w = _body_pos(robot, robot_cfg) - robot.data.root_pos_w
    return _finite_clip(quat_apply_inverse(robot.data.root_quat_w, rel_pos_w), 4.0)


def object_to_wrist_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object-to-right-wrist vector, expressed in robot-root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    rel_pos_w = obj.data.root_pos_w - _body_pos(robot, robot_cfg)
    return _finite_clip(quat_apply_inverse(robot.data.root_quat_w, rel_pos_w), 4.0)


def target_position_b(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Place target position relative to the robot root, expressed in robot-root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    rel_pos_w = _target_pos_w(env, target_pos) - robot.data.root_pos_w
    return _finite_clip(quat_apply_inverse(robot.data.root_quat_w, rel_pos_w), 4.0)


def object_to_target(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object-to-place-target vector in world frame."""
    obj: RigidObject = env.scene[object_cfg.name]
    rel_pos_w = obj.data.root_pos_w - _target_pos_w(env, target_pos)
    return _finite_clip(rel_pos_w, 4.0)


def object_height(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object root height above world origin."""
    obj: RigidObject = env.scene[object_cfg.name]
    return _finite_clip(obj.data.root_pos_w[:, 2:3], 4.0)
