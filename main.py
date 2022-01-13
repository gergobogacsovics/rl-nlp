from envs.basic_env import NLPEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import pandas as pd
from util_tensorboard import TensorboardLoggerSimple
import numpy as np

CONFIG = {
    "num_envs": 1
}


def make_env(seed):
    def _f():
        env = NLPEnv(data_path="../BloombergNRG_train_rl.csv", window_size=5, last_index=episode_length,
                     logger=TensorboardLoggerSimple(log_dir="ppo_tensorboard"))
        env.seed(seed)
        return env

    return _f


def make_val_env(seed, train_env):
    def _f():
        env = NLPEnv(data_path="../BloombergNRG_val.csv", window_size=5, last_index=episode_length, env_type="validation",
                     logger=TensorboardLoggerSimple(log_dir="ppo_tensorboard", run_id=train_env.logger.run_id))
        env.seed(seed)
        return env

    return _f


if __name__ == '__main__':
    df = pd.read_csv("../BloombergNRG.csv", sep=";")

    # df["direction"] = np.sign(df["roberta_large_score"])

    episode_length = df.shape[0] * 0.8

    # env = NLPEnv(data_path="../BloombergNRG.csv", window_size=5, last_index=episode_length,
    #             logger=TensorboardLoggerSimple(log_dir="ppo_tensorboard"))

    env = DummyVecEnv([make_env(seed=thread) for thread in range(CONFIG["num_envs"])])  # DummyVecEnv([lambda: env])
    val_env = DummyVecEnv([make_val_env(seed=0, train_env=env)])
    # action_space_size = env.action_space.n
    # state_space_size = env.observation_space.n

    ppo_args = {
        # "batch_size":
        "learning_rate": 3e-4,
        "batch_size": 128,  # 32,
        "n_steps": int(episode_length * 10),  # 30 int(episode_length * 3),  # buffer size
        "n_epochs": 3
    }

    # MlpLstmPolicy
    model = PPO("MlpPolicy", env, verbose=0, device="cpu", **ppo_args)
    # model.policy.to("cpu")
    model.learn(total_timesteps=episode_length * 10000, eval_env=val_env, eval_freq=episode_length, n_eval_episodes=1)  # 10000

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()