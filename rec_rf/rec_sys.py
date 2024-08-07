import gym
from gym import spaces
import random

import csv
from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from collections import defaultdict
from ray import air
from ray import tune
from tqdm import tqdm
from ray.rllib.algorithms.ppo import PPOConfig
import os
import shutil
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms import ppo
import ray
from ray import tune
import gym

# import gymnasium as gym


class JokeRec(gym.Env):
    def __init__(self, config):
        # NB: here we're passing strings via config; RLlib use of JSON
        # parser was throwing exceptions due to config values
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
        """
        attempt a warm start, sampling half the dense submatrix of most-rated items
        """
        sample_size = round(len(self.dense) / 2.0)

        for action, items in self.clusters.items():
            for item in random.sample(self.dense, sample_size):
                if item in items:
                    state, reward, done, info = self.step(action)

    def _get_state(self):
        """
        calculate root-mean-square (i.e., normalized vector distance) for the agent's current
        "distance" measure from each cluster center as the observation
        """
        n = float(len(self.used))

        if n > 0.0:
            state = [np.sqrt(x / n) for x in self.coords]
        else:
            state = self.coords

        return state

    def reset(self):
        """
        reset the item recommendation history, select a new user to simulate from among the
        dataset rows, then run an initial 'warm-start' sequence of steps before handing step
        control back to the agent
        """
        self.count = 0
        self.used = []
        self.depleted = 0
        self.coords = [np.float64(0.0)] * K_CLUSTERS

        # select a random user to simulate
        self.data_row = random.choice(self.dataset)

        # attempt a warm start
        self._warm_start()

        return self._get_state()

    def step(self, action):
        """
        recommend one item, which may result in a no-op --
        in production, skip any repeated items per user
        """
        assert action in self.action_space, action
        assert_info = "c[item] {}, rating {}, scaled_diff {}"

        # enumerate items from the cluster which is selected by the
        # action, which in turn haven't been recommended previously
        # to the simulated user
        items = set(self.clusters[action]).difference(set(self.used))

        if len(items) < 1:
            # oops! items from the selected cluster have been
            # depleted, i.e. all have been recommended previously to
            # the simulated user; hopefully the agent will learn to
            # switch to exploring among the other clusters
            self.depleted += 1
            item = None
            reward = REWARD_DEPLETED
        else:
            # chose an item at random from the selected cluster
            item = random.choice(tuple(items))
            rating = self.data_row[item]

            if not rating:
                # no-op! this action resulted in an unrated item
                reward = REWARD_UNRATED

            else:
                # success! this action resulted in an item rated by the simulated user
                reward = rating
                self.used.append(item)

                # update the coords history: agent observes its distance to each cluster "evolve"
                for i in range(len(self.coords)):
                    c = self.centers[i]

                    # note that `rating` values are already scaled,
                    # and the `c[item]` cluster center also is an
                    # average of the scaled ratings for a given item
                    scaled_diff = abs(c[item] - rating) / 2.0
                    self.coords[i] += scaled_diff**2.0

        self.count += 1
        done = self.count >= MAX_STEPS
        info = {"item": item, "count": self.count, "depleted": self.depleted}

        return self._get_state(), reward, done, info

    def render(self, mode="human"):
        last_used = self.used[-10:]
        last_used.reverse()
        print(">> used:", last_used)
        print(">> dist:", [round(x, 2) for x in self._get_state()])
        print(">> depl:", self.depleted)


NO_RATING = "99"
MAX_RATING = 10.0


def load_data(data_path):
    rows = []

    with open(data_path, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")

        for row in csvreader:
            conv = [None] * (len(row) - 1)

            for i in range(1, len(row)):
                if row[i] != NO_RATING:
                    rating = float(row[i]) / MAX_RATING
                    conv[i - 1] = rating

            rows.append(conv)

    return rows


def get_scaled_clusters(centers):
    """
    return a DataFrame with the item-to-center "distance" scaled
    within `[0.0, 1.0]` where `0.0` represents "nearest"
    """
    df_scaled = pd.DataFrame()

    df = pd.DataFrame(centers)
    n_items = df.shape[1]

    for item in range(n_items):
        row = df[item].values
        item_max = max(row)
        item_min = min(row)
        scale = item_max - item_min

        df_scaled[item] = pd.Series([1.0 - (val - item_min) / scale for val in row])

    return df_scaled


def partition_items(df):
    """
    return a partitioned map, where each cluster is a set sampled
    from its "nearest" items
    """
    k = df.shape[0]
    n_items = df.shape[1]

    clusters = defaultdict(set)
    selected = set()
    i = 0

    while len(selected) < n_items:
        label = i % k
        i += 1

        row = df.loc[label, :].values.tolist()

        gradient = {item: dist for item, dist in enumerate(row) if item not in selected}

        nearest_item = min(gradient, key=gradient.get)
        selected.add(nearest_item)
        clusters[label].add(nearest_item)

    return clusters


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


DENSE_SUBMATRIX = [5, 7, 8, 13, 15, 16, 17, 18, 19, 20]
centers = km.cluster_centers_
df_scaled = get_scaled_clusters(centers)
clusters = partition_items(df_scaled)

CENTERS = centers.tolist()
CLUSTERS = dict(clusters)


CONFIG = PPOConfig().copy()

CONFIG["log_level"] = "WARN"
CONFIG["num_workers"] = 0  # set to `0` for debug

CONFIG["env_config"] = {
    "dataset": DATA_PATH,
    "dense": str(DENSE_SUBMATRIX),
    "clusters": repr(CLUSTERS),
    "centers": repr(CENTERS),
}

ROW_LENGTH = 100
MAX_STEPS = ROW_LENGTH - len(DENSE_SUBMATRIX)

REWARD_DEPLETED = -0.1  # item recommended from a depleted cluster (no-op)
REWARD_UNRATED = -0.05  # item was never rated by this user

env = JokeRec(CONFIG["env_config"])
env.reset()

action = env.action_space.sample()
print("action:", action)

state, reward, done, info = env.step(action)
print("obs:", state)
print("reward:", reward)


print("baseline result>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


def run_one_episode(env, naive=False, verbose=False):
    """
    step through one episode, using either a naive strategy or random actions
    """
    env.reset()
    sum_reward = 0

    action = None
    avoid_actions = set([])
    depleted = 0

    for i in range(MAX_STEPS):
        if not naive or not action:
            action = env.action_space.sample()

        state, reward, done, info = env.step(action)

        if verbose:
            print("action:", action)
            print("obs:", i, state, reward, done, info)

        # naive strategy: select items from the nearest non-depleted cluster
        if naive:
            if info["depleted"] > depleted:
                depleted = info["depleted"]
                avoid_actions.add(action)

            obs = []

            for a in range(len(state)):
                if a not in avoid_actions:
                    dist = round(state[a], 2)
                    obs.append([dist, a])

            action = min(obs)[1]

        sum_reward += reward

        if done:
            if verbose:
                print("DONE @ step {}".format(i))

            break

    if verbose:
        print("CUMULATIVE REWARD: ", round(sum_reward, 3))

    return sum_reward


def measure_baseline(env, n_iter=1, naive=False, verbose=False):
    history = []

    for episode in tqdm(range(n_iter), ascii=True, desc="measure baseline"):
        sum_reward = run_one_episode(env, naive=naive, verbose=verbose)
        history.append(sum_reward)

    baseline = sum(history) / len(history)
    return baseline


baseline = measure_baseline(env, n_iter=1000, naive=True)
print("BASELINE CUMULATIVE REWARD", round(baseline, 3))


print(
    "run ray and rlib>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
)


# CHECKPOINT_ROOT = "tmp/rec"
# shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

# ray_results = "{}/ray_results/".format(os.getenv("HOME"))
# shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

# info = ray.init(ignore_reinit_error=True, num_cpus=1)


import os
import shutil
import ray
from ray.rllib.algorithms import ppo

CHECKPOINT_ROOT = "tmp/rec"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

ray_results = "{}/ray_results/".format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)
info = ray.init(ignore_reinit_error=True, num_cpus=1)


lo = np.array([np.float64(-1.0)] * K_CLUSTERS)
hi = np.array([np.float64(1.0)] * K_CLUSTERS)
observation_space = spaces.Box(
            lo, hi, shape=(K_CLUSTERS,), dtype=np.float64
        )
action_space = spaces.Discrete(K_CLUSTERS)
CONFIG = PPOConfig()

CONFIG["log_level"] = "WARN"
CONFIG["num_workers"] = 1

CONFIG["env_config"] = {
    "dataset": DATA_PATH,
    "dense": str(DENSE_SUBMATRIX),
    "clusters": repr(CLUSTERS),
    "centers": repr(CENTERS),
    "observation_space": observation_space,
}


def create_joke_rec_env(config):
    return JokeRec(CONFIG["env_config"])
    # return JokeRec()


env_key = "JokeRec-v0"
# register_env(env_key, create_joke_rec_env)
# register_env(env_key, lambda config_env: JokeRec(config_env))
# AGENT = ppo.PPO(env=env_key,config=CONFIG)

config = {
    # "env": create_joke_rec_env,
    "framework": "torch",
    "num_gpus": 0,
    "lr": 5e-5,
}

trainer = ppo.PPO(config=config)
# trainer = ppo.PPO(config=config, env="CartPole-v1")


# def create_joke_rec_env(config):
#     return JokeRec(CONFIG["env_config"])

# register_env("joke_rec", create_joke_rec_env)
# ray.init()
# algo = ppo.PPO(env="joke_rec")

# config = PPOConfig()
# config = CONFIG
# config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size=128)
# config = config.resources(num_gpus=0)
# config = config.env_runners(num_env_runners=1)

# config.training(lr=tune.grid_search([0.001]), clip_param=0.2)
# config = config.environment(env="joke_rec")
# tune.Tuner(
#     "PPO",
#     run_config=air.RunConfig(stop={"training_iteration": 1}),
#     param_space=config.to_dict(),
# ).fit()

# # 初始化Ray
# ray.init()

# env_key = "JokeRec-v0"
# config_env = CONFIG["env_config"]
# # register_env(env_key, lambda config_env: JokeRec(config_env))
# # AGENT = (
# #      CONFIG
# #     .environment(env=env_key)
# #     # .framework(framework="torch")
# #     .training(train_batch_size=1000, sgd_minibatch_size=56, num_sgd_iter=10)
# #     # .rollouts(num_rollout_workers=1)
# #     # .resources(num_gpus=0)
# #     .build()
# # )

# stop = {
#     "training_iteration": 10000,
# }

# trainer = ppo.PPOTrainer(config=CONFIG, env=env_key)
# for i in range(stop["training_iteration"]):
#     result = trainer.train()
#     print(result)
