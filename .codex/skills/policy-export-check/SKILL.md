---
name: policy-export-check
description: Export and validate RSL-RL policies for sim2sim and sim2real from the g1-loco-manipulation project.
---

# Policy Export Check Skill

Use for sim2sim, sim2real, TorchScript/JIT export, ONNX export, and deployment validation.

## File distinction

```text
model_xxxx.pt          # RSL-RL checkpoint for runner restore
exported/policy.pt     # TorchScript/JIT policy
exported/policy.onnx   # ONNX policy
```

## RSL-RL 5.x export

Prefer runner-native export methods:

```python
export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
```

## Validation checklist

After export, validate:

1. Input observation dimension.
2. Output action dimension.
3. JIT/ONNX output matches `runner.get_inference_policy()` on the same observation.
4. Observation ordering matches training.
5. Action ordering matches trained joints.
6. Action scale and default joint offset are replicated in deployment.
7. Control frequency matches or is compensated.
8. PD gains match deployment assumptions.
9. IMU/base velocity coordinate frames are consistent.
10. Safety clipping exists on the real robot side.

Known prior G1 setup printed:

```text
input observation dimension: 480
output action dimension: 29
```

Do not hard-code this blindly; verify against the current model.
