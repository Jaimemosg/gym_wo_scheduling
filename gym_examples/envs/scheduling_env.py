# %% Environment for scheduling work orders
from datetime import datetime, timedelta
from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
from gymnasium.spaces import Box
from pydantic import BaseModel, ValidationError


class WorkOrdersDict(BaseModel):
    required_techs: list[int]
    resource_availability: list[bool]
    fis: list[float]
    job_duration: list[float]


class ScheduleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        n_techs=5,
        work_order_df=pd.DataFrame(),
        start_date="2024-08-26",
    ):
        self.total_fis = work_order_df["fis"].sum()
        self.total_time = work_order_df["job_duration"].sum()
        n_jobs = len(work_order_df)
        self.n_jobs = n_jobs
        self.n_techs = n_techs
        self.state = None
        self.n_available_techs = None
        self.schedule = None

        # Check data integrity
        data: dict = work_order_df.to_dict(orient="list")
        try:
            WorkOrdersDict(**data)
            # item = WorkOrdersDict(**data)
            # print(item)
        except ValidationError as e:
            print("Validation error:", e)

        # Store work orders in a dictionary
        ids = list(range(n_jobs))
        records = work_order_df.to_dict(orient="records")
        self.work_orders_dict = dict(zip(ids, records))

        # Create date attributes
        date_format = "%Y-%m-%d %H:%M:%S"
        start_date = start_date + " 08:00:00"
        self.init_date = datetime.strptime(start_date, date_format)
        self.end_date = self.init_date + timedelta(days=5, hours=8)

        self.action_space = Box(low=0, high=1, shape=(self.n_jobs,), dtype=int)

        """
        A matrix with the following attributes for each work order:
            - Techs required (proportion of total available per week)
            - Resource availability (yes/no)
            - FIS (proportion of total FIS)
            - Job duration (proportion of total time, both in minutes)
            - Assigned work order (yes/no)
        """
        self.n_att = 5
        self.observation_space = Box(
            low=0, high=1, shape=(self.n_jobs, self.n_att), dtype=float
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return self.schedule

    def _update_schedule_tracker(self):
        for job, job_details in self.schedule.items():
            if job_details:
                if self.time > job_details[1] and not job_details[3]:
                    self.n_available_techs += job_details[2]
                    self.schedule[job][3] = True  # Mark as completed
                    self.available_techs = np.concat(
                        self.available_techs, job_details[3]
                    )

    def _compute_reward(self, action: np.ndarray) -> None | int:
        reward = 0
        n_available_techs = self.n_available_techs
        available_techs = self.available_techs

        if action.sum() == 0:  # Don't schedule any work orders
            """
                The agent will be greedy. No reward if no work orders are scheduled 
                when there are available techs.
            """
            if n_available_techs == 0:
                reward = self.total_fis  # / self.n_jobs
        else:
            all_states = []
            for i in range(len(action)):
                # State (job) attributes
                required_techs = self.work_orders_dict[i]["required_techs"]
                job_duration = self.work_orders_dict[i]["job_duration"]
                fis = self.work_orders_dict[i]["fis"]
                resource_availability = self.work_orders_dict[i][
                    "resource_availability"
                ]
                state_i = [
                    required_techs / self.n_techs,
                    resource_availability,
                    fis / self.total_fis,
                    job_duration / self.total_time,
                ]

                if action[i] == 1:
                    state_i.append(1)  # Assigned work order

                    if not self.schedule[i]:
                        employed_techs = n_available_techs - required_techs

                        if employed_techs < 0:
                            """
                                If there are not techs enough and the agent assigns a 
                                work order, penalize.
                            """
                            reward -= self.total_fis
                        else:
                            employed_techs = np.random.choice(
                                a=n_available_techs,
                                size=required_techs,
                                replace=False,
                            )
                            n_available_techs -= required_techs
                            available_techs = np.setdiff1d(
                                available_techs, employed_techs
                            )
                            self.available_techs = available_techs
                            self.n_available_techs = n_available_techs

                            self.schedule[i] = (
                                self.time,  # Start time
                                self.time + timedelta(minutes=job_duration),  # End time
                                required_techs,  # Required techs
                                employed_techs,  # Employed techs
                                # False,  # Mark as not completed
                            )
                            reward += fis
                    else:
                        """If order i exists and the agent assigns again, penalize."""
                        reward -= self.total_fis
                else:
                    state_i.append(0)  # Not assigned work order

                all_states.append(state_i)

            self.state = np.array(all_states).reshape(self.n_jobs, self.n_att)
        return reward

    def _increase_time_step(self):
        self.time += timedelta(minutes=30)

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, dict]:
        self._update_schedule_tracker()
        reward = self._compute_reward(action)
        self._increase_time_step()
        terminated = self.time > self.end_date
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros((self.n_jobs, self.n_att))

        # Reset schedule
        self.schedule = {i: None for i in range(self.n_jobs)}

        # Set techs available
        self.n_available_techs = self.n_techs
        self.available_techs = np.array(list(range(self.n_techs)))

        # Reset time
        self.time = self.init_date

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def render(self):
        # if self.render_mode == "rgb_array":
        return self._render_frame()

    def _render_frame(self):
        df = []
        for i in range(self.n_jobs):  # Size of space of actions
            if self.schedule[i]:
                for tech in range(self.n_techs):
                    employed_techs = self.schedule[i][3]  # employed_techs
                    if tech in employed_techs:
                        instance = dict()
                        instance["Start"] = self.schedule[i][0]
                        instance["Finish"] = self.schedule[i][1]
                        instance["Task"] = f"Tech {tech}"
                        instance["Job"] = f"Work Order {i}"
                        df.append(instance)

        fig = None
        if len(df) > 0:
            df = pd.DataFrame(df)
            fig = ff.create_gantt(
                df,
                index_col="Job",
                show_colorbar=True,
                group_tasks=True,
            )
            fig.update_yaxes(
                autorange="reversed"
            )  # otherwise tasks are listed from the bottom up
        return fig

    def close(self):
        pass
