# Unitree RL Lab (Fork)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![rsl-rl](https://img.shields.io/badge/rsl_rl-5.0.1-blue)](https://github.com/leggedrobotics/rsl_rl)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

基于 [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) 的修改版本，适配 **rsl-rl-lib 5.x** 并修复 G1 机器人的加载问题。

## Overview

Reinforcement learning training and deployment for Unitree robots (Go2, Go2W, B2, H1, H1_2, G1-23dof, G1-29dof). Built on NVIDIA IsaacLab/IsaacSim with RSL-RL for PPO policy training. Policies are trained in simulation, exported as ONNX, and deployed to real robots via a C++ controller.

<div align="center">

| <div align="center"> Isaac Lab </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif" width="240px">](g1_sim.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](g1_mujoco.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](g1_real.gif) |

</div>

## Changes from Upstream

相比原版 [unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)，本 fork 做了以下修改：

### 1. G1 机器人切换为 URDF 加载方式

原版 G1 使用 USD 模型加载，但 USD 文件中 link prims 缺少 `UsdPhysics.RigidBodyAPI`，导致 IsaacLab 创建场景时抛出：

```
ValueError: No contact sensors added to the prim: '/World/envs/env_0/Robot'.
This means that no rigid bodies are present under this prim.
```

本 fork 将 G1-23dof、G1-29dof、G1-29dof-Mimic 三个配置从 `UnitreeUsdFileCfg` 切换为 `UnitreeUrdfFileCfg`，使用 URDF 方式加载（即官方 README 中的 Method 2）。

### 2. 适配 rsl-rl-lib 5.x

原版代码针对 rsl-rl-lib 2.3.x，与 5.x 存在不兼容：

- `KeyError: 'class_name'` — 5.x 要求 `actor`/`critic` 字段替代旧的 `policy` 字段
- `TypeError: ... unexpected keyword argument 'stochastic'` — 5.x 的 `MLPModel` 不接受废弃参数

修改内容：
- 两个 agent 配置 (`locomotion/agents/rsl_rl_ppo_cfg.py`, `mimic/agents/rsl_rl_ppo_cfg.py`) 从 `RslRlPpoActorCriticCfg` 迁移到 `RslRlMLPModelCfg`
- `train.py` 和 `play.py` 中添加字典过滤，移除 5.x 不接受的废弃字段

### Modified Files

| File | Change |
|------|--------|
| `source/.../assets/robots/unitree.py` | G1 configs: USD -> URDF; set `UNITREE_ROS_DIR` |
| `source/.../tasks/locomotion/agents/rsl_rl_ppo_cfg.py` | Adapt to rsl-rl 5.x actor/critic format |
| `source/.../tasks/mimic/agents/rsl_rl_ppo_cfg.py` | Same as above |
| `scripts/rsl_rl/train.py` | Filter deprecated config fields for rsl-rl 5.x |
| `scripts/rsl_rl/play.py` | Same as above |

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- Install the Unitree RL IsaacLab standalone environments.

  - Clone this repository separately from the Isaac Lab installation:

    ```bash
    git clone https://github.com/54muzi/rl_push.git
    ```
  - Use a python interpreter that has Isaac Lab installed, install the library in editable mode:

    ```bash
    conda activate env_isaaclab
    ./unitree_rl_lab.sh -i
    # restart your shell to activate the environment changes.
    ```
- Download unitree robot description files

  *Method 1: Using USD Files* (Go2, H1, B2)
  - Download unitree usd files from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main)
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - Config `UNITREE_MODEL_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.

  *Method 2: Using URDF Files [Required for G1]* Only for IsaacSim >= 5.0
  - Download unitree robot urdf files from [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
    ```bash
    git clone https://github.com/unitreerobotics/unitree_ros.git
    ```
  - Config `UNITREE_ROS_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.

- Verify that the environments are correctly installed:

  ```bash
  ./unitree_rl_lab.sh -l
  ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity
  ```

## Usage

```bash
# Train a locomotion policy
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity
# or: python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity

# Play/export a trained checkpoint (exports ONNX + JIT)
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity

# Resume interrupted training
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity --resume

# List all registered environments
./unitree_rl_lab.sh -l

# Build C++ deploy controller
cd deploy/robots/g1_29dof && mkdir build && cd build && cmake .. && make

# Lint
pre-commit run --all-files

# Monitor training with TensorBoard
tensorboard --logdir logs/rsl_rl/
```

## Architecture

### Data flow

`IsaacSim training -> ONNX + deploy.yaml export -> C++ controller loads model -> real robot via unitree_sdk2 (DDS)`

### Python side (`source/unitree_rl_lab/`)

- **`assets/robots/unitree.py`** — Articulation configs for all robots (USD/URDF paths, actuator params, joint mappings).
- **`tasks/locomotion/`** — Velocity-tracking MDP per robot. Each robot has `velocity_env_cfg.py` defining scene, observations, actions, rewards, terminations, curriculum, events. Shared MDP logic in `mdp/` subdirs.
- **`tasks/mimic/`** — Motion-mimicry tasks (G1-29dof only: dance_102, gangnam_style).

### Scripts (`scripts/rsl_rl/`)

- **`train.py`** — Entry point using `@hydra_task_config`. Creates Gym env, wraps for RSL-RL, runs `OnPolicyRunner.learn()`.
- **`play.py`** — Loads checkpoint, runs inference, exports policy as ONNX/JIT.

### C++ deploy (`deploy/`)

- FSM: `Passive -> FixStand -> RLBase` (and `Mimic` for G1).
- `include/` — Shared headers: FSM framework, ONNX inference wrapper.
- `robots/<robot>/` — Per-robot `main.cpp`, `config/config.yaml`. Each has its own `CMakeLists.txt`.
- Runtime deps: Boost, yaml-cpp, Eigen3, spdlog, fmt, unitree_sdk2, onnxruntime 1.22.0.

## Deploy

### Setup

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2 && mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF && sudo make install
cd unitree_rl_lab/deploy/robots/g1_29dof && mkdir build && cd build && cmake .. && make
```

### Sim2Sim (Mujoco)

Install [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation), then:

```bash
cd unitree_mujoco/simulate/build && ./unitree_mujoco
cd unitree_rl_lab/deploy/robots/g1_29dof/build && ./g1_ctrl
# 1. Press [L2 + Up] to stand up
# 2. Click mujoco window, press 8 to touch ground
# 3. Press [R1 + X] to run policy
# 4. Click mujoco window, press 9 to disable elastic band
```

### Sim2Real

```bash
./g1_ctrl --network eth0
```

## Acknowledgements

- [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab): Original project by Unitree Robotics
- [IsaacLab](https://github.com/isaac-sim/IsaacLab): Foundation for training and running codes
- [mujoco](https://github.com/google-deepmind/mujoco): Simulation functionalities
- [robot_lab](https://github.com/fan-ziqi/robot_lab): Referenced for project structure
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): Versatile humanoid control framework
