# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Unitree RL Lab — reinforcement learning training and deployment for Unitree robots (Go2, Go2W, B2, H1, H1_2, G1-23dof, G1-29dof). Built on NVIDIA IsaacLab/IsaacSim with RSL-RL for PPO policy training. Policies are trained in simulation, exported as ONNX, and deployed to real robots via a C++ controller.

## Commands

```bash
# Install (requires conda env env_isaaclab)
conda activate env_isaaclab
./unitree_rl_lab.sh -i          # pip install -e source/unitree_rl_lab/

# Train a locomotion policy
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity
# or: python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity

# Play/export a trained checkpoint (exports ONNX + JIT)
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity

# List all registered environments
./unitree_rl_lab.sh -l

# Build C++ deploy controller (per robot)
cd deploy/robots/g1_29dof && mkdir build && cd build && cmake .. && make

# Lint
pre-commit run --all-files
```

## Architecture

### Data flow

`IsaacSim training → ONNX + deploy.yaml export → C++ controller loads model → real robot via unitree_sdk2 (DDS)`

### Python side (`source/unitree_rl_lab/`)

- **`assets/robots/unitree.py`** — Articulation configs for all robots (USD paths, actuator params, joint mappings). `unitree_actuators.py` defines custom actuator models.
- **`tasks/locomotion/`** — Velocity-tracking MDP per robot. Each robot dir has `velocity_env_cfg.py` defining scene, observations, actions, rewards, terminations, curriculum, events. Shared MDP logic lives in `mdp/` subdirs: `observations.py`, `rewards.py`, `commands/`, `curriculums.py`.
- **`tasks/mimic/`** — Motion-mimicry tasks (G1-29dof only: dance_102, gangnam_style).
- **`tasks/__init__.py`** — Auto-registers all task envs via `import_packages`.

### Scripts (`scripts/rsl_rl/`)

- **`train.py`** — Entry point using `@hydra_task_config`. Creates Gym env, wraps for RSL-RL, runs `OnPolicyRunner.learn()`. Also exports `deploy.yaml` at training start.
- **`play.py`** — Loads checkpoint, runs inference, exports policy as ONNX/JIT.
- **`cli_args.py`** — Argument parsing shared by train/play.

### C++ deploy (`deploy/`)

- FSM: `Passive → FixStand → RLBase` (and `Mimic` for G1).
- `include/` — Shared headers: FSM framework, ONNX inference wrapper, IsaacLab algorithm reimplementations in C++.
- `robots/<robot>/` — Per-robot `main.cpp`, `config/config.yaml`, compiled policy .cpp files. Each has its own `CMakeLists.txt`.
- Runtime deps: Boost, yaml-cpp, Eigen3, spdlog, fmt, unitree_sdk2, onnxruntime 1.22.0.

### Adding a new robot

1. Add articulation config in `assets/robots/unitree.py`
2. Create `tasks/locomotion/robots/<robot>/velocity_env_cfg.py` with env config
3. Register in `tasks/locomotion/<robot>/__init__.py`
4. Add C++ controller under `deploy/robots/<robot>/`

## Code conventions

- Python linting: black, flake8, isort (configured in `pyproject.toml` and `.flake8`)
- No test suite exists in this repository
- Chinese comments are used throughout MDP/reward/env files to explain RL reward shaping logic
- Environment task names follow the pattern `Unitree-<Robot>-Velocity` (locomotion) or `Unitree-Robot-v0` (mimic)
