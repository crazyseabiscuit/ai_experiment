import gymnasium as gym
from tqdm import tqdm
from gymnasium import spaces
import os

import ray
from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np

# from ray.rllib.algorithms.ppo import PPO
# import ray.rllib.algorithms.ppo as ppo
from collections import defaultdict
from pathlib import Path
from utils import load_data
import pandas as pd
import random

from utils import get_scaled_clusters, partition_items, load_data
from sklearn.impute import SimpleImputer

from ray.rllib.algorithms.ppo import PPOConfig
from collections import defaultdict
from sklearn.cluster import KMeans
import os

info = ray.init(ignore_reinit_error=True, num_cpus=1)

NO_RATING = "99"
MAX_RATING = 10.0
ROW_LENGTH = 100
DENSE_SUBMATRIX = [5, 7, 8, 13, 15, 16, 17, 18, 19, 20]
MAX_STEPS = ROW_LENGTH - len(DENSE_SUBMATRIX)

REWARD_DEPLETED = -0.1
REWARD_UNRATED = -0.05

K_CLUSTERS = 7


class JokeRec(gym.Env):
    def __init__(self, config):
        self.dense = eval(config["dense"])
        self.centers = eval(config["centers"])
        self.clusters = eval(config["clusters"])

        lo = np.array([np.float64(-1.0)] * K_CLUSTERS)
        hi = np.array([np.float64(1.0)] * K_CLUSTERS)

        self.observation_space = spaces.Box(
            lo, hi, shape=(K_CLUSTERS,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(K_CLUSTERS)

        self.dataset = load_data(config["dataset"])

    def _warm_start(self):
        sample_size = round(len(self.dense) / 2.0)

        for action, items in self.clusters.items():
            for item in random.sample(self.dense, sample_size):
                if item in items:
                    state, reward, done, info, _ = self.step(action)

    def _get_state(self):
        n = float(len(self.used))

        if n > 0.0:
            state = [np.sqrt(x / n) for x in self.coords]
        else:
            state = self.coords

        return state

    def reset(self, *, seed=None, options=None):
        self.count = 0
        self.used = []
        self.depleted = 0
        self.coords = [np.float64(0.0)] * K_CLUSTERS

        self.data_row = random.choice(self.dataset)

        self._warm_start()

        return self._get_state(), {}

    def step(self, action):
        assert action in self.action_space, action
        assert_info = "c[item] {}, rating {}, scaled_diff {}"

        items = set(self.clusters[action]).difference(set(self.used))

        if len(items) < 1:
            self.depleted += 1
            item = None
            reward = REWARD_DEPLETED
        else:
            item = random.choice(tuple(items))
            rating = self.data_row[item]

            if not rating:
                reward = REWARD_UNRATED

            else:
                reward = rating
                self.used.append(item)

                for i in range(len(self.coords)):
                    c = self.centers[i]
                    scaled_diff = abs(c[item] - rating) / 2.0
                    self.coords[i] += scaled_diff**2.0

        self.count += 1
        done = self.count >= MAX_STEPS
        info = {"item": item, "count": self.count, "depleted": self.depleted}

        return self._get_state(), reward, done, info, {}

    def render(self, mode="human"):
        last_used = self.used[-10:]
        last_used.reverse()
        print(">> used:", last_used)
        print(">> dist:", [round(x, 2) for x in self._get_state()])
        print(">> depl:", self.depleted)


def process_cluster():
    DATA_PATH = Path(os.getcwd()) / Path("jester-data-1.csv")
    sample = load_data(DATA_PATH)
    df = pd.DataFrame(sample)
    print(df.head())

    imp = SimpleImputer(missing_values=np.nan, strategy="median")
    imp.fit(df.values)

    X = imp.transform(df.values)

    K_CLUSTERS = 7
    km = KMeans(n_clusters=K_CLUSTERS)
    km.fit(X)
    y_km = km.fit_predict(X)

    centers = km.cluster_centers_
    df_scaled = get_scaled_clusters(centers)
    clusters = partition_items(df_scaled)
    return DATA_PATH, centers, clusters


DATA_PATH, centers, clusters = process_cluster()

CENTERS = centers.tolist()
CLUSTERS = dict(clusters)

CONFIG = PPOConfig().copy()

CONFIG["log_level"] = "WARN"
CONFIG["num_workers"] = 1  # set to `0` for debug

CONFIG["env_config"] = {
    "dataset": DATA_PATH,
    "dense": str(DENSE_SUBMATRIX),
    "clusters": repr(CLUSTERS),
    "centers": repr(CENTERS),
}


joke_env = JokeRec(CONFIG["env_config"])
joke_env.reset()
df = pd.DataFrame(columns=["reward_min", "reward_mean", "reward_max", "checkpoint"])
status = "Episode Reward Min: {:.3f}, Episode Reward Mean: {:.3f}, Episode Reward Max: {:.3f}, Checkpoint: {}"

config = PPOConfig().environment(JokeRec, env_config=CONFIG["env_config"])
config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size=128)
config = config.resources(num_gpus=0)
config = config.env_runners(num_env_runners=1)

CHECKPOINT_ROOT = "tmp/rec"
print("build algorithm ********************************")
algo = config.build()
print("start algorithm ********************************")
for _ in tqdm(range(10)):
    result = algo.train()
    checkpoint_file = algo.save(CHECKPOINT_ROOT)
    row = [
        result["env_runners"]["episode_reward_min"],
        result["env_runners"]["episode_reward_mean"],
        result["env_runners"]["episode_reward_max"],
        checkpoint_file,
    ]
    df.loc[len(df)] = row
    print(status.format(*row))


ray.shutdown()
