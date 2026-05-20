---
name: safe-git-update
description: Safely commit and push g1-loco-manipulation project changes without uploading logs, model assets, checkpoints, or external repositories.
---

# Safe Git Update Skill

Repository root:

```bash
/home/xiao/g1-loco-manipulation
```

## Never commit

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

## Checks

```bash
cd /home/xiao/g1-loco-manipulation
git status
git diff
git status --short
```

If generated files are staged:

```bash
git reset unitree_ros logs outputs
```

If already tracked:

```bash
git rm -r --cached unitree_ros logs outputs
```

## Stage specific files

```bash
git add source
git add scripts/train.py
git add scripts/play.py
git add scripts/export_policy.py
git add scripts/list_tasks.py
git add .gitignore
git add README.md AGENTS.md ACKNOWLEDGEMENTS.md
```

## Commit examples

```bash
git commit -m "Fix RSL-RL 5.x play compatibility"
git commit -m "Add clipped action-rate reward"
git commit -m "Stabilize G1 training action regularization"
git commit -m "Add push cart v1 asset"
```

## Push

```bash
git push origin main
```

Prefer SSH remote:

```bash
git remote set-url origin git@github.com:54muzi/g1-loco-manipulation.git
ssh -T git@github.com
git push origin main
```

## Expected output

When preparing a commit, report:

1. Files safe to stage.
2. Files that must not be staged.
3. Recommended commit message.
4. Exact commit and push commands.
