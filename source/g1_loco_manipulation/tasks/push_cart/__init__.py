"""Gym registration for the G1 push-cart loco-manipulation task."""

import gymnasium as gym

gym.register(
    id="G1-PushCart-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_push_cart_env_cfg:PushCartEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.g1_push_cart_env_cfg:PushCartPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "g1_loco_manipulation.tasks.push_cart.agents.rsl_rl_ppo_cfg:PushCartPPORunnerCfg",
    },
)
