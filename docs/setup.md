# Setup

Install Isaac Sim and Isaac Lab following the upstream Isaac Lab documentation, then install this package in editable mode:

```bash
cd /home/xiao/g1-loco-manipulation
python -m pip install -e .
```

Download Unitree robot descriptions separately. This local project defaults to the ignored robot asset bundle under:

```text
source/g1_loco_manipulation/assets/robots/unitree_ros
source/g1_loco_manipulation/assets/robots/g1-29dof_wholebody_inspire
```

For a custom location, override the default per command:

```bash
UNITREE_ROS_DIR=/path/to/unitree_ros python scripts/list_tasks.py
```
