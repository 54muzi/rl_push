# G1 Loco-Manipulation

Minimal research code for Unitree G1 locomotion and incremental loco-manipulation experiments in Isaac Lab.

This repository is intentionally smaller than the development sandbox it was extracted from. It currently contains a G1 29-DoF velocity-tracking baseline, a first push-cart loco-manipulation task skeleton, and a fixed-base G1 Inspire pick-place task.

## Quick Checks

```bash
python scripts/list_tasks.py
python scripts/train.py --task Debug-G1 --num_envs 2 --max_iterations 1 --headless
python scripts/train.py --task G1-Locomotion-Velocity --num_envs 16 --max_iterations 1 --headless
python scripts/train.py --task G1-PushCart-Velocity-v0 --num_envs 16 --max_iterations 1 --headless
python scripts/train.py --task G1-PickPlace-Inspire-v0 --num_envs 4 --max_iterations 1 --headless
```

If your Isaac Lab session does not default to the desired GPU, pass `--device cuda:0` or another explicit device.

## External Assets

Robot description assets are not redistributed in git. By default, the code reads the local ignored robot asset bundle next to `g1.py`:

```text
source/g1_loco_manipulation/assets/robots/unitree_ros
source/g1_loco_manipulation/assets/robots/g1-29dof_wholebody_inspire
```

If your assets are elsewhere, override the defaults per command:

```bash
UNITREE_ROS_DIR=/path/to/unitree_ros python scripts/train.py --task G1-Locomotion-Velocity
```

The current G1 URDF baseline resolves:

```text
source/g1_loco_manipulation/assets/robots/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf
```

The G1 Inspire USD asset resolves:

```text
source/g1_loco_manipulation/assets/robots/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0_no_hand_camera.usd
```

The push-cart task uses an in-repository lightweight URDF asset:

```text
source/g1_loco_manipulation/assets/objects/service_cart/urdf/service_cart_v1.urdf
```

The cart is v1.0 scaffolding: a box body with four fixed wheel cylinders and a passive prismatic joint that constrains motion along the cart's local X axis. It is intended for stable first contact experiments, not realistic caster or wheel dynamics.

## Tasks

Registered tasks:

```text
Debug-G1
G1-Locomotion-Velocity
G1-PickPlace-Inspire-v0
G1-PickPlace-Inspire-v0-Play
G1-PushCart-Velocity-v0
```

Run the fixed-base G1 debug task:

```bash
python scripts/train.py --task Debug-G1 --num_envs 2 --max_iterations 1 --headless
```

`Debug-G1` uses a DearPyGui joint controller. If the GUI dependency is not installed:

```bash
python -m pip install -e ".[gui]"
```

Run it interactively with a display:

```bash
python scripts/train.py --task Debug-G1 --num_envs 2 --max_iterations 100000 --device cuda:0
```

Use `G1_DEBUG_GUI=0` for headless smoke tests that keep the GUI action term but do not open the controller window:

```bash
G1_DEBUG_GUI=0 python scripts/train.py --task Debug-G1 --num_envs 2 --max_iterations 1 --headless
```

Train the push-cart task:

```bash
python scripts/train.py \
  --task G1-PushCart-Velocity-v0 \
  --num_envs 16 \
  --max_iterations 1 \
  --headless
```

Train the fixed-base G1 Inspire pick-place task:

```bash
python scripts/train.py \
  --task G1-PickPlace-Inspire-v0 \
  --num_envs 4 \
  --max_iterations 1 \
  --headless
```

This task uses the local G1 Inspire USD, a primitive table, and a primitive cube. It is a first runnable migration stage of the WBC-AGILE pick-place workflow, not the full WBC tracking-imitation setup with external motion data and teacher policies.

Play a push-cart checkpoint:

```bash
python scripts/play.py \
  --task G1-PushCart-Velocity-v0 \
  --checkpoint /path/to/logs/rsl_rl/g1_push_cart_velocity/<run>/model_<iter>.pt \
  --num_envs 1 \
  --real-time
```

The `--task` argument must match the checkpoint. A push-cart checkpoint expects the push-cart observation shape; playing it with the default locomotion task will fail with a model input-size mismatch.

## Scope

Included now:

- Fixed-base `Debug-G1` task for robot asset and RSL-RL smoke testing.
- G1 29-DoF velocity locomotion baseline.
- G1 push-cart velocity task skeleton.
- Fixed-base G1 Inspire pick-place task skeleton.
- Minimal rail-constrained service cart URDF asset.
- RSL-RL 5.x compatibility through Isaac Lab's `handle_deprecated_rsl_rl_cfg`.
- Train, play, task listing, and policy export scripts.
- Minimal MDP helpers needed by the baseline and push-cart task.

Intentionally not included:

- Unitree robot asset repositories.
- Training logs, checkpoints, exported policies, or wandb data.
- Go2, H1, B2, mimic, Docker, MuJoCo, and physical deployment code.
- Real robot network configuration.
- WBC-AGILE motion files, teacher policies, and remote YCB/table assets.
- Real caster dynamics, tactile sensing, dexterous hand modeling, or sim2real push-cart deployment.

## Acknowledgements

This code is built upon Unitree RL Lab, NVIDIA Isaac Lab, and RSL-RL. See `ACKNOWLEDGEMENTS.md` for attribution and asset notes.
