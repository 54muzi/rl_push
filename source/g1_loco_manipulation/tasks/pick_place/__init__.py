"""Gym registration for G1 Inspire pick-place tasks."""

import gymnasium as gym


gym.register(
    id="G1-PickPlace-Inspire-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_inspire_pick_place_env_cfg:PickPlaceEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.g1_inspire_pick_place_env_cfg:PickPlacePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
    },
)


gym.register(
    id="G1-PickPlace-Inspire-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_inspire_pick_place_env_cfg:PickPlacePlayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.g1_inspire_pick_place_env_cfg:PickPlacePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
    },
)
