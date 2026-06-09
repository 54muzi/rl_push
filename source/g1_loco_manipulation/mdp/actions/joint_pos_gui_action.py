"""Interactive joint-position action for debugging G1 joints."""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions import JointPositionAction

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import JointPositionGUIActionCfg


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=10)
def _resolve_g1_lr_joint_indices(joint_names: tuple[str, ...]) -> tuple[list[int], list[int]]:
    """Return left-right remap indices and sign-flip indices for G1-style joint names."""

    mirrored_indices = []
    for joint_name in joint_names:
        if "left" in joint_name:
            mirrored_joint_name = joint_name.replace("left", "right")
        elif "right" in joint_name:
            mirrored_joint_name = joint_name.replace("right", "left")
        else:
            mirrored_joint_name = joint_name

        if mirrored_joint_name not in joint_names:
            raise ValueError(f"Mirrored joint name '{mirrored_joint_name}' not found in action joints.")
        mirrored_indices.append(joint_names.index(mirrored_joint_name))

    neg_indices = [
        index
        for index, joint_name in enumerate(joint_names)
        if any(token in joint_name for token in ("roll", "yaw", "hand")) and "thumb_0" not in joint_name
    ]
    return mirrored_indices, neg_indices


def _mirror_g1_joint_targets(targets: torch.Tensor, joint_names: tuple[str, ...]) -> torch.Tensor:
    mirrored_indices, neg_indices = _resolve_g1_lr_joint_indices(joint_names)
    mirrored_targets = targets.clone()
    mirrored_targets[..., mirrored_indices] = targets
    mirrored_targets[..., neg_indices] *= -1.0
    return mirrored_targets


class JointPositionGUIAction(JointPositionAction):
    """Joint-position action controlled by a DearPyGui slider window.

    Incoming policy actions are ignored. All simulated environments receive the same
    GUI targets, except odd-numbered environments can receive the left-right mirrored
    target when ``mirror_actions`` is enabled.
    """

    cfg: JointPositionGUIActionCfg

    def __init__(self, cfg: JointPositionGUIActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._lock = threading.Lock()
        self._desired_pos = self._select_joints(self._asset.data.joint_pos).clone()
        self._desired_stiffness, self._desired_damping = self._build_selected_gains()
        self._default_stiffness = self._desired_stiffness.clone()
        self._default_damping = self._desired_damping.clone()
        self._mirror_actions = cfg.mirror_actions
        self._joint_ids_tensor = self._joint_ids_as_tensor()
        self._actuator_joint_ids = {
            name: torch.as_tensor(self._asset.find_joints(actuator.joint_names)[0], device=self.device, dtype=torch.long)
            for name, actuator in self._asset.actuators.items()
        }
        self._full_stiffness_buffer = torch.zeros(self.num_envs, self._asset.num_joints, device=self.device)
        self._full_damping_buffer = torch.zeros_like(self._full_stiffness_buffer)
        self._gui_thread: threading.Thread | None = None

        launch_gui = cfg.launch_gui and not _env_flag("G1_DEBUG_GUI_DISABLE") and os.environ.get("G1_DEBUG_GUI") != "0"
        if launch_gui:
            try:
                import dearpygui.dearpygui  # noqa: F401
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Debug-G1 GUI requires dearpygui. Install it in the Isaac Lab Python environment with "
                    "`/home/xiao/miniconda3/envs/env_isaaclab/bin/python -m pip install dearpygui`, "
                    "or run with `G1_DEBUG_GUI=0` to disable the GUI."
                ) from exc
            self._gui_thread = threading.Thread(target=self._launch_gui, name="JointPositionGUI", daemon=True)
            self._gui_thread.start()

    def process_actions(self, actions: torch.Tensor) -> None:  # noqa: ARG002
        """Ignore policy actions; the GUI state is authoritative."""
        return None

    def apply_actions(self) -> None:
        """Apply GUI joint targets and GUI-selected PD gains."""
        with self._lock:
            target_pos = self._desired_pos.clone().to(device=self.device)
            target_stiffness = self._desired_stiffness.clone().to(device=self.device)
            target_damping = self._desired_damping.clone().to(device=self.device)

        if self._mirror_actions and self.num_envs > 1:
            mirrored_pos = _mirror_g1_joint_targets(target_pos, tuple(self._joint_names))
            odd_env_mask = (torch.arange(target_pos.shape[0], device=self.device) % 2 == 1).unsqueeze(1)
            target_pos = torch.where(odd_env_mask, mirrored_pos, target_pos)

        self._asset.set_joint_position_target(target_pos, joint_ids=self._joint_ids)
        self._apply_selected_gains(target_stiffness, target_damping)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            contains_first_env = True
        elif torch.is_tensor(env_ids):
            contains_first_env = bool(torch.any(env_ids == 0).item())
        else:
            contains_first_env = 0 in env_ids
        if env_ids is None or contains_first_env:
            with self._lock:
                self._desired_pos[:] = self._select_joints(self._asset.data.joint_pos).clone()
                self._desired_stiffness[:] = self._default_stiffness
                self._desired_damping[:] = self._default_damping

    def _launch_gui(self) -> None:
        import dearpygui.dearpygui as dpg

        dpg.create_context()
        dpg.create_viewport(title="Debug-G1 Joint Controller", width=640, height=920)

        pos_slider_tags: list[int | str] = []
        stiffness_slider_tags: list[int | str] = []
        damping_slider_tags: list[int | str] = []
        effort_bar_tags: list[int | str] = []

        with dpg.window(label="Joint Position Controller", width=620, height=860, pos=(10, 10)):
            dpg.add_text("Debug-G1 joint targets")
            dpg.add_separator()

            def _reset_joints_cb() -> None:
                with self._lock:
                    self._desired_pos[:] = self._select_joints(self._asset.data.default_joint_pos).clone()
                    self._desired_stiffness[:] = self._default_stiffness
                    self._desired_damping[:] = self._default_damping
                    desired_pos = self._desired_pos[0].detach().cpu()
                    desired_stiffness = self._desired_stiffness[0].detach().cpu()
                    desired_damping = self._desired_damping[0].detach().cpu()
                for idx in range(len(self._joint_names)):
                    dpg.set_value(pos_slider_tags[idx], float(desired_pos[idx]))
                    dpg.set_value(stiffness_slider_tags[idx], float(desired_stiffness[idx]))
                    dpg.set_value(damping_slider_tags[idx], float(desired_damping[idx]))

            def _randomize_joints_cb() -> None:
                with self._lock:
                    limits = self._selected_joint_limits().detach().cpu()
                    low = limits[:, 0]
                    high = limits[:, 1]
                    random_pos = low + (high - low) * torch.rand_like(low)
                    self._desired_pos[:] = random_pos.to(device=self.device).unsqueeze(0)
                for idx in range(len(self._joint_names)):
                    dpg.set_value(pos_slider_tags[idx], float(random_pos[idx]))

            with dpg.group(horizontal=True):
                dpg.add_button(label="Reset", callback=_reset_joints_cb)
                dpg.add_button(label="Randomize", callback=_randomize_joints_cb)
            dpg.add_separator()

            with dpg.theme() as effort_bar_theme:
                with dpg.theme_component(dpg.mvStemSeries):
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 5, category=dpg.mvThemeCat_Plots)

            for local_id, joint_name in enumerate(self._joint_names):
                joint_idx = local_id if isinstance(self._joint_ids, slice) else self._joint_ids[local_id]
                limits = self._asset.data.soft_joint_pos_limits[0, joint_idx].detach().cpu()
                low, high = float(limits[0]), float(limits[1])
                current_pos = float(self._desired_pos[0, local_id].detach().cpu())
                current_stiffness = float(self._desired_stiffness[0, local_id].detach().cpu())
                current_damping = float(self._desired_damping[0, local_id].detach().cpu())
                effort_limit = self._joint_effort_limit(joint_idx)

                with dpg.group(horizontal=True):
                    with dpg.group():
                        pos_slider_tags.append(
                            dpg.add_slider_float(
                                label=f"[{local_id:02d}] {joint_name}",
                                min_value=low,
                                max_value=high,
                                default_value=current_pos,
                                callback=self._make_pos_callback(local_id),
                                format="%.3f",
                                width=360,
                            )
                        )
                        stiffness_slider_tags.append(
                            dpg.add_slider_float(
                                label="P",
                                min_value=0.0,
                                max_value=self.cfg.max_stiffness,
                                default_value=current_stiffness,
                                callback=self._make_stiffness_callback(local_id),
                                format="%.1f",
                                indent=20,
                                width=320,
                            )
                        )
                        damping_slider_tags.append(
                            dpg.add_slider_float(
                                label="D",
                                min_value=0.0,
                                max_value=self.cfg.max_damping,
                                default_value=current_damping,
                                callback=self._make_damping_callback(local_id),
                                format="%.1f",
                                indent=20,
                                width=320,
                            )
                        )
                    with dpg.plot(no_title=True, no_menus=True, no_box_select=True, no_mouse_pos=True, height=88, width=60):
                        dpg.add_plot_axis(dpg.mvXAxis, no_gridlines=True, no_tick_marks=True, no_tick_labels=True)
                        dpg.set_axis_limits(dpg.last_item(), -1.0, 1.0)
                        with dpg.plot_axis(
                            dpg.mvYAxis, no_gridlines=True, no_tick_marks=True, no_tick_labels=True
                        ) as y_axis:
                            dpg.set_axis_limits(y_axis, -effort_limit, effort_limit)
                            effort_bar = dpg.add_stem_series([0.0], [0.0])
                            dpg.bind_item_theme(effort_bar, effort_bar_theme)
                            effort_bar_tags.append(effort_bar)
                dpg.add_separator()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            applied_effort = self._select_joints(self._asset.data.applied_torque).detach().cpu()
            for idx in range(len(self._joint_names)):
                dpg.set_value(effort_bar_tags[idx], [[0.0], [float(applied_effort[0, idx])]])
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    def _make_pos_callback(self, index: int):
        def _callback(sender, app_data, user_data=None) -> None:  # noqa: ARG001
            with self._lock:
                self._desired_pos[:, index] = float(app_data)

        return _callback

    def _make_stiffness_callback(self, index: int):
        def _callback(sender, app_data, user_data=None) -> None:  # noqa: ARG001
            with self._lock:
                self._desired_stiffness[:, index] = float(app_data)

        return _callback

    def _make_damping_callback(self, index: int):
        def _callback(sender, app_data, user_data=None) -> None:  # noqa: ARG001
            with self._lock:
                self._desired_damping[:, index] = float(app_data)

        return _callback

    def _build_selected_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        full_stiffness = torch.zeros(self.num_envs, self._asset.num_joints, device=self.device)
        full_damping = torch.zeros_like(full_stiffness)
        for actuator in self._asset.actuators.values():
            joint_ids = self._asset.find_joints(actuator.joint_names)[0]
            full_stiffness[:, joint_ids] = actuator.stiffness
            full_damping[:, joint_ids] = actuator.damping
        return self._select_joints(full_stiffness).clone(), self._select_joints(full_damping).clone()

    def _apply_selected_gains(self, selected_stiffness: torch.Tensor, selected_damping: torch.Tensor) -> None:
        for name, actuator in self._asset.actuators.items():
            joint_ids = self._actuator_joint_ids[name]
            self._full_stiffness_buffer[:, joint_ids] = actuator.stiffness
            self._full_damping_buffer[:, joint_ids] = actuator.damping

        self._full_stiffness_buffer[:, self._joint_ids_tensor] = selected_stiffness
        self._full_damping_buffer[:, self._joint_ids_tensor] = selected_damping

        for name, actuator in self._asset.actuators.items():
            joint_ids = self._actuator_joint_ids[name]
            actuator.stiffness[:] = self._full_stiffness_buffer.index_select(1, joint_ids)
            actuator.damping[:] = self._full_damping_buffer.index_select(1, joint_ids)

    def _joint_ids_as_tensor(self) -> torch.Tensor:
        if isinstance(self._joint_ids, slice):
            stop = self._joint_ids.stop or self._asset.num_joints
            return torch.arange(self._joint_ids.start or 0, stop, self._joint_ids.step or 1, device=self.device)
        return torch.as_tensor(self._joint_ids, dtype=torch.long, device=self.device)

    def _select_joints(self, tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(self._joint_ids, slice):
            slicer = [slice(None)] * tensor.ndim
            slicer[1] = self._joint_ids
            return tensor[tuple(slicer)]
        return tensor.index_select(1, torch.as_tensor(self._joint_ids, dtype=torch.long, device=tensor.device))

    def _selected_joint_limits(self) -> torch.Tensor:
        limits = self._asset.data.soft_joint_pos_limits[0]
        if isinstance(self._joint_ids, slice):
            return limits[self._joint_ids]
        return limits.index_select(0, torch.as_tensor(self._joint_ids, dtype=torch.long, device=limits.device))

    def _joint_effort_limit(self, joint_idx: int) -> float:
        for actuator in self._asset.actuators.values():
            actuator_joint_ids = self._asset.find_joints(actuator.joint_names)[0]
            if joint_idx in actuator_joint_ids:
                local_idx = actuator_joint_ids.index(joint_idx)
                return float(actuator.effort_limit[0, local_idx].detach().cpu())
        return 1.0
