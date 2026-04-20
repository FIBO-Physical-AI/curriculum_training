import gymnasium as gym

gym.register(
    id="Curriculum-Go2-Velocity-Uniform",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_velocity_uniform:UniformCurriculumEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.go2_velocity_uniform:UniformCurriculumPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Curriculum-Go2-Velocity-TaskSpec",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_velocity_task_specific:TaskSpecificCurriculumEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.go2_velocity_task_specific:TaskSpecificCurriculumPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Curriculum-Go2-Velocity-Teacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_velocity_teacher_guided:TeacherGuidedCurriculumEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.go2_velocity_teacher_guided:TeacherGuidedCurriculumPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
