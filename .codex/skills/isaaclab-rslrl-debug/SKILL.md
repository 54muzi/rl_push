---
name: isaaclab-rslrl-debug
description: Debug Isaac Lab + RSL-RL training, play, checkpoint loading, and policy export problems in the g1-loco-manipulation project.
---

# Isaac Lab / RSL-RL Debug Skill

Use this skill for:

- `G1-Locomotion-Velocity`
- Isaac Lab task registration
- RSL-RL 5.x compatibility
- `train.py` / `play.py`
- checkpoint loading
- policy export
- PPO instability
- action-rate explosion
- `std >= 0.0` errors

## 1. Classify the failure

Identify whether the problem happens during:

```text
A. import / task registration
B. env creation
C. training rollout
D. PPO update
E. checkpoint loading
F. play inference
G. JIT/ONNX export
```

## 2. Common import failure

If the error is:

```text
ModuleNotFoundError: No module named 'omni'
```

check whether `SimulationApp` has started before imports that touch Isaac Lab or `isaaclab_rl`.

Move heavy imports after:

```python
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

## 3. RSL-RL config compatibility

If the error is `KeyError: 'class_name'`, use:

```python
import importlib.metadata as metadata
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

installed_version = metadata.version("rsl-rl-lib")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
```

Pass the version string, not a `Version` object.

## 4. RSL-RL 5.x play inference

Do not use:

```python
runner.alg.policy
runner.alg.actor_critic
```

Use:

```python
policy = runner.get_inference_policy(device=env.unwrapped.device)
```

Then:

```python
with torch.inference_mode():
    actions = policy(obs)
```

## 5. RSL-RL 5.x export

Prefer:

```python
runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
```

## 6. std checkpoint mismatch

If loading reports:

```text
Missing key(s): distribution.log_std_param
Unexpected key(s): distribution.std_param
```

then the checkpoint was trained with scalar std but the current config expects log std.

Use the same std config for play as was used for training, or retrain with the new std config. Avoid `strict=False` except for temporary deterministic playback tests.

## 7. PPO instability diagnosis

If error is:

```text
RuntimeError: normal expects all elements of std >= 0.0
```

inspect:

```text
Loss/value
Policy/mean_std
Episode_Reward/action_rate
Train/mean_reward
Train/mean_episode_length
```

Likely chain:

```text
action outlier → action_rate spike → return/value target spike → value loss explosion → actor distribution instability
```

Recommended mitigations:

- Add action clip.
- Clamp action-rate reward.
- Reduce action-rate penalty weight.
- Lower learning rate.
- Consider log std or std clamp for new runs.
- Add checks for obs/action NaN or extreme values.

## Expected output

When applying a fix, report:

1. Files changed.
2. Root cause.
3. Exact change made.
4. Smoke-test command.
5. Old checkpoint compatibility impact.
