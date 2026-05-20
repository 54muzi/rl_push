# G1 Loco-Manipulation

Minimal research code for Unitree G1 locomotion and future loco-manipulation experiments in Isaac Lab.

This repository is intentionally smaller than the development sandbox it was extracted from. The first runnable target is a G1 29-DoF velocity-tracking baseline:

```bash
python scripts/list_tasks.py
python scripts/train.py --task G1-Locomotion-Velocity --num_envs 16 --max_iterations 1 --headless
```

## External Assets

Robot description assets are not redistributed in this repository. By default, the code reads the existing sandbox asset paths:

```text
/home/xiao/g1-loco-manipulation/unitree_ros
```

If your assets are elsewhere, override the defaults per command:

```bash
UNITREE_ROS_DIR=/path/to/unitree_ros python scripts/train.py --task G1-Locomotion-Velocity
```

The current G1 baseline resolves:

```text
unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf
```

## Scope

Included now:

- G1 29-DoF velocity locomotion baseline.
- RSL-RL 5.x compatibility through Isaac Lab's `handle_deprecated_rsl_rl_cfg`.
- Train, play, task listing, and policy export scripts.
- Minimal MDP helpers needed by the baseline.

Intentionally not included:

- Unitree robot asset repositories.
- Training logs, checkpoints, exported policies, or wandb data.
- Go2, H1, B2, mimic, Docker, MuJoCo, and physical deployment code.
- Real robot network configuration.

## Acknowledgements

This code is built upon Unitree RL Lab, NVIDIA Isaac Lab, and RSL-RL. See `ACKNOWLEDGEMENTS.md` for attribution and asset notes.
