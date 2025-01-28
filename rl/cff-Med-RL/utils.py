from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import Dict
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
import numpy as np
from ray.tune.logger import LoggerCallback
import os
import shutil


class actionCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == -1, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        episode.user_data["action"] = []
        episode.hist_data["actions"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length >= 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        actions = episode._agent_collectors[ "agent0" ].buffers["actions"]
        episode.hist_data["actions"] = list(np.array(actions).flat)


class srcLoggerCallback(LoggerCallback):
    def __init__(self, file_paths: list = []):
        self._file_paths = file_paths

    def log_trial_start(self, trial):
        for f in self._file_paths:
            file_name = f.split("/")[-1]
            trial_file = os.path.join(trial.local_path, file_name)# for ray 2.38.0
            shutil.copy2(f, trial_file)
