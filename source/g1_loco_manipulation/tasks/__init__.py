"""Gym task registration for this project.

Task modules are imported explicitly so lightweight commands such as
``scripts/list_tasks.py`` do not pull in Isaac Lab modules that require
SimulationApp.
"""

from .debug import *  # noqa: F401, F403
from .locomotion import *  # noqa: F401, F403
from .pick_place import *  # noqa: F401, F403
from .push_cart import *  # noqa: F401, F403
