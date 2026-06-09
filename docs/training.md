# Training

List registered tasks:

```bash
python scripts/list_tasks.py
```

Run a one-iteration smoke test:

```bash
G1_DEBUG_GUI=0 python scripts/train.py --task Debug-G1 --num_envs 2 --max_iterations 1 --headless
python scripts/train.py --task G1-Locomotion-Velocity --num_envs 16 --max_iterations 1 --headless
python scripts/train.py --task G1-PushCart-Velocity-v0 --num_envs 16 --max_iterations 1 --headless
python scripts/train.py --task G1-PickPlace-Inspire-v0 --num_envs 4 --max_iterations 1 --headless
```

Run the fixed-base G1 GUI joint debugger:

```bash
python -m pip install -e ".[gui]"
python scripts/train.py --task Debug-G1 --num_envs 2 --max_iterations 100000 --device cuda:0
```

Run the first G1 Inspire pick-place migration:

```bash
python scripts/train.py --task G1-PickPlace-Inspire-v0 --num_envs 4 --max_iterations 1 --headless
```

`G1-PickPlace-Inspire-v0` is fixed-base and self-contained: it uses the local G1 Inspire USD, a primitive table, a primitive cube, and right arm plus right Inspire hand joint-position actions.
