from gymnasium.envs.registration import register

register(
    id="ScheduleEnv-v0",
    entry_point="gym_examples.envs:ScheduleEnv",
)
