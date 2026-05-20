# Setup

Install Isaac Sim and Isaac Lab following the upstream Isaac Lab documentation, then install this package in editable mode:

```bash
cd /home/xiao/g1-loco-manipulation
python -m pip install -e .
```

Download Unitree robot descriptions separately. This local project defaults to:

```text
/home/xiao/g1-loco-manipulation/unitree_ros
```

For a custom location, override the default per command:

```bash
UNITREE_ROS_DIR=/path/to/unitree_ros python scripts/list_tasks.py
```
