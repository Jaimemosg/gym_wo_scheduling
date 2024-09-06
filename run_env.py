# %% Create environment
import gymnasium
import numpy as np
import pandas as pd

import gym_examples  # noqa

"""
Columns:
- fis
- job_duration
- required_techs
- resource_availability
"""
n = 30
n_techs = 5
sim_fis = np.random.normal(5000, 200, n)
job_duration = np.random.poisson(60, n)
required_techs = np.random.randint(1, n_techs, n)
resorce_availability = np.random.choice([1, 0], n)
df = pd.DataFrame(
    [
        {
            "fis": sim_fis[i],
            "job_duration": job_duration[i],
            "required_techs": required_techs[i],
            "resource_availability": resorce_availability[i],
        }
        for i in range(n)
    ]
)

env = gymnasium.make(
    "ScheduleEnv-v0",
    render_mode="human",
    n_techs=n_techs,
    work_order_df=df,
)
observation, info = env.reset()

# %% Run
for _ in range(2):
    action = (
        env.action_space.sample()
    )  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break  # observation, info = env.reset()

    if env.render():
        env.render().to_image()

env.close()
