import gymnasium as gym
import numpy as np
import wandb
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from base_env import BaseEnv


class PPOWrapper(gym.ObservationWrapper):
    """Wrap BaseEnv to return encoded observation as numeric vector."""

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def observation(self, obs):
        encoded_obs = self.env.encode_obs(obs)
        return encoded_obs


def make_env(seed=None):
    env = BaseEnv(seed=seed)
    env = PPOWrapper(env)
    return env


class WandbCallback(BaseCallback):
    """Log episode metrics and training time to W&B."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = None

    def _on_training_start(self) -> None:
        """Record start time when training begins."""
        self.start_time = time.time()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        elapsed_time = time.time() - self.start_time if self.start_time else 0.0

        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                reward = ep_info.get("r", 0)
                length = ep_info.get("l", 0)

                env_info = info.get("env_info", {})
                iou = env_info.get("iou", 0.0)
                cosine_sim = env_info.get("cosine_sim", 0.0)
                span_start = env_info.get("span_start", 0)
                span_end = env_info.get("span_end", 0)

                wandb.log({
                    "episode/reward": reward,
                    "episode/length": length,
                    "episode/iou": iou,
                    "episode/cosine_sim": cosine_sim,
                    "episode/span_start": span_start,
                    "episode/span_end": span_end,
                    "step": self.num_timesteps,
                    "elapsed_time_sec": elapsed_time
                })
        return True


if __name__ == "__main__":
    # 1. Initialize W&B
    wandb.init(
        project="RL_final_project",
        name="ppo_sb3_run1",
        config={
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "batch_size": 64,
            "gamma": 0.99,
            "clip_range": 0.2,
            "net_arch": [512, 512],
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )

    # 2. Create environment
    env = make_env(seed=42)
    check_env(env, warn=True)

    # 3. Configure SB3 logger to W&B
    tmp_path = "./sb3_logs"
    new_logger = configure(tmp_path, ["stdout", "tensorboard"])

    # 4. Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=64,
        n_steps=2048,
        learning_rate=3e-4,
        ent_coef=0.01,
        gamma=0.99,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[512, 512])
    )
    model.set_logger(new_logger)

    # 5. Train PPO with W&B callback
    total_timesteps = 50000
    model.learn(total_timesteps=total_timesteps, callback=WandbCallback())

    # 6. Save the model
    model.save("ppo_grounded_text")
    wandb.finish()
