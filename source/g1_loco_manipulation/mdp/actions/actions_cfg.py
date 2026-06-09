"""Configuration objects for custom action terms."""

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .joint_pos_gui_action import JointPositionGUIAction


@configclass
class JointPositionGUIActionCfg(isaac_mdp.JointActionCfg):
    """Joint-position action whose targets are controlled from a DearPyGui window."""

    class_type: type[ActionTerm] = JointPositionGUIAction

    use_default_offset: bool = True
    """Use the articulation default joint positions as the action offset."""

    max_stiffness: float = 200.0
    """Maximum P-gain shown by the GUI slider."""

    max_damping: float = 25.0
    """Maximum D-gain shown by the GUI slider."""

    mirror_actions: bool = True
    """Apply a left-right mirrored command to odd-numbered environments."""

    launch_gui: bool = True
    """Launch the GUI window when the action term is constructed."""
