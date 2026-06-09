# Troubleshooting

If task registration fails, first check that the package can be imported:

```bash
python scripts/list_tasks.py
```

If G1 asset loading fails, first check the configured asset path:

```bash
grep -n "UNITREE_ROS_DIR" source/g1_loco_manipulation/assets/robots/g1.py
ls source/g1_loco_manipulation/assets/robots/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf
ls source/g1_loco_manipulation/assets/robots/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0_no_hand_camera.usd
```

For a non-standard asset location, override the default per command:

```bash
UNITREE_ROS_DIR=/path/to/unitree_ros python scripts/train.py --task G1-Locomotion-Velocity --headless
```

For RSL-RL 5.x config compatibility, this repository uses Isaac Lab's `handle_deprecated_rsl_rl_cfg`.
