# Bug 修复总结

## 问题描述

运行 `./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity` 时遇到两个连续的错误。

---

## Bug 1: G1 USD 模型缺少 RigidBodyAPI

### 错误信息

```
ValueError: No contact sensors added to the prim: '/World/envs/env_0/Robot'.
This means that no rigid bodies are present under this prim. Please check the prim path.
```

### 根本原因

G1 29dof 的 USD 模型文件 (`unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd`) 中，link prims 缺少 `UsdPhysics.RigidBodyAPI`。IsaacLab 在创建场景时需要遍历 Robot prim 下所有 link，为带有 RigidBodyAPI 的 link 添加 contact sensor。由于 G1 USD 中没有此 API，`activate_contact_sensors` 函数找不到任何 rigid body，抛出异常。

### 解决方案

将 G1 系列机器人从 USD 加载方式切换为 URDF 加载方式（README 中的 Method 2，官方推荐方式）。IsaacLab 的 URDF 导入器会自动正确处理 RigidBodyAPI。

### 修改的文件

#### 1. `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`

- 设置 `UNITREE_ROS_DIR` 指向克隆的 unitree_ros 仓库
- G1-29dof、G1-23dof、G1-29dof-Mimic 三个配置从 `UnitreeUsdFileCfg` 改为 `UnitreeUrdfFileCfg`

```python
# 之前
UNITREE_ROS_DIR = "/home/xiao/rl_push/unitree_ros"  # Replace with ...

# 之后
UNITREE_ROS_DIR = "/home/xiao/rl_push/unitree_ros"
```

```python
# 之前
UNITREE_G1_29DOF_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
    ),
    ...
)

# 之后
UNITREE_G1_29DOF_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=f"{UNITREE_ROS_DIR}/robots/g1_description/g1_29dof_rev_1_0.urdf",
    ),
    ...
)
```

`UNITREE_G1_23DOF_CFG` 和 `UNITREE_G1_29DOF_MIMIC_CFG` 做了相同修改。

#### 额外操作

- 克隆了 `https://github.com/unitreerobotics/unitree_ros.git` 到项目根目录
- 删除了尝试修复 USD 的临时脚本 `scripts/fix_g1_usd.py`

---

## Bug 2: RSL-RL 5.x 与旧版配置格式不兼容

### 错误信息（两轮）

第一轮：
```
KeyError: 'class_name'
```

第二轮（修复第一轮后）：
```
TypeError: MLPModel.__init__() got an unexpected keyword argument 'stochastic'
```

### 根本原因

环境中安装的 `rsl-rl-lib` 版本为 **5.0.1**，而项目代码编写时针对的是 **2.3.x**。RSL-RL 4.0+ 做了以下不兼容变更：

1. **废弃 `policy` 字段，改用 `actor` + `critic`**：旧版 `RslRlPpoActorCriticCfg` 已废弃，新版使用 `RslRlMLPModelCfg`
2. **`MLPModel.__init__()` 参数变化**：新版只接受 `hidden_dims`, `activation`, `obs_normalization`, `distribution_cfg`，不接受废弃的 `stochastic`, `init_noise_std`, `noise_std_type` 等
3. **需要 `obs_groups` 映射**：新版需要显式指定 observation groups 到 actor/critic 的映射
4. **`empirical_normalization` 废弃**：改用 model 级别的 `obs_normalization`

### 解决方案

三管齐下：更新配置类格式 + 显式设置废弃字段 + 在运行时过滤废弃字段。

### 修改的文件

#### 2. `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`

```python
# 之前 (rsl-rl 2.x 格式)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = ...

# 之后 (rsl-rl 5.x 格式)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        stochastic=False,
        init_noise_std=1.0,
        obs_normalization=False,
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=None,
        stochastic=False,
        init_noise_std=1.0,
        obs_normalization=False,
    )
    algorithm = ...
```

关键变更：
- 导入 `RslRlMLPModelCfg` 替代 `RslRlPpoActorCriticCfg`
- `policy=` 拆分为 `actor=` + `critic=`
- 移除 `empirical_normalization`，添加 `obs_groups`
- actor 使用 `distribution_cfg=GaussianDistributionCfg(...)` 替代 `init_noise_std`
- critic 使用 `distribution_cfg=None`（确定性输出）
- 显式设置 `stochastic=False`, `init_noise_std=1.0` 覆盖 `MISSING` 默认值

#### 3. `source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/agents/rsl_rl_ppo_cfg.py`

与 locomotion agent 相同的修改（`entropy_coef` 保留为 0.005）。

#### 4. `scripts/rsl_rl/train.py`

在 `OnPolicyRunner` 创建前添加 dict 过滤，移除 RSL-RL 5.x 不接受的废弃字段：

```python
def _filter_model_cfg(d: dict) -> dict:
    keep = {"hidden_dims", "activation", "obs_normalization", "distribution_cfg",
            "class_name", "cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"}
    return {k: v for k, v in d.items() if k in keep}

agent_dict = agent_cfg.to_dict()
for key in ("actor", "critic"):
    if key in agent_dict and isinstance(agent_dict[key], dict):
        agent_dict[key] = _filter_model_cfg(agent_dict[key])
runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)
```

#### 5. `scripts/rsl_rl/play.py`

同样的过滤逻辑添加到 play 脚本的 runner 创建处。

---

## 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `source/.../assets/robots/unitree.py` | G1 配置从 USD 切换到 URDF；设置 UNITREE_ROS_DIR |
| `source/.../tasks/locomotion/agents/rsl_rl_ppo_cfg.py` | 适配 rsl-rl 5.x 的 actor/critic 配置格式 |
| `source/.../tasks/mimic/agents/rsl_rl_ppo_cfg.py` | 同上 |
| `scripts/rsl_rl/train.py` | 添加 `_filter_model_cfg` 过滤废弃字段 |
| `scripts/rsl_rl/play.py` | 同上 |

## 环境依赖

- 额外克隆了 `unitree_ros` 仓库（URDF 模型文件来源）
- 已安装 `rsl-rl-lib==5.0.1`（不需要降级，代码已适配）
- IsaacSim 5.1.0 + IsaacLab 2.3.0
