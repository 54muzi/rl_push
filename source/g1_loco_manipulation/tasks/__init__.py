"""Gym task registration for this project.

Task modules are imported explicitly so lightweight commands such as
``scripts/list_tasks.py`` do not pull in Isaac Lab modules that require
SimulationApp.
"""

from .locomotion import *  # noqa: F401, F403
