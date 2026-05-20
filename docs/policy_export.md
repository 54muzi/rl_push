# Policy Export

Export a trained RSL-RL checkpoint:

```bash
python scripts/export_policy.py \
  --task G1-Locomotion-Velocity \
  --checkpoint /path/to/model.pt
```

Expected outputs:

```text
exported/policy.pt
exported/policy.onnx
```

