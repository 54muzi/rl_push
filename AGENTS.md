# AGENTS.md - G1 Loco-Manipulation

This repository is the clean public-code track for Unitree G1 locomotion and future loco-manipulation work.

## Scope

- Keep `/home/xiao/rl_push` as the development sandbox.
- Keep this repository focused on the G1 research line.
- The first runnable target is `G1-Locomotion-Velocity`.
- Push-cart and contact-aware manipulation should be added incrementally after the locomotion baseline is stable.

## Hard Rules

1. Do not commit generated training outputs.
2. Do not commit external asset repositories.
3. Do not commit checkpoints or exported policies.
4. Keep Unitree assets out of git.
5. The default asset paths live in `source/g1_loco_manipulation/assets/robots/g1.py`; environment variables are optional overrides.
6. Do not silently change reward, observation, termination, or action semantics.
7. Keep changes small and verify registration/import behavior before broader migration.

## Never Commit

```text
unitree_ros/
logs/
outputs/
runs/
wandb/
checkpoints/
*.pt
*.pth
*.onnx
```

## Smoke Tests

```bash
python scripts/list_tasks.py
python scripts/train.py --task G1-Locomotion-Velocity --num_envs 16 --max_iterations 1 --headless
```

If the train smoke test cannot find the G1 URDF, override per command:

```bash
UNITREE_ROS_DIR=/path/to/unitree_ros python scripts/train.py --task G1-Locomotion-Velocity --headless
```
