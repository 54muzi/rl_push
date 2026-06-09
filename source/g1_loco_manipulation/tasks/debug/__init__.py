"""Gym registration for lightweight G1 debug tasks."""

import gymnasium as gym


gym.register(
    id="Debug-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.debug_env_cfg:G1DebugEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.debug_env_cfg:G1DebugEnvCfg",
        "rsl_rl_cfg_entry_point": "g1_loco_manipulation.tasks.debug.agents.rsl_rl_ppo_cfg:DebugPPORunnerCfg",
    },
)
