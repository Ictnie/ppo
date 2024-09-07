
import gymnasium as gym
import os
# from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Retrieve the model from the hub
## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename = name of the model zip file from the repository
os.environ["TRUST_REMOTE_CODE"] = "True"

# checkpoint=load_from_hub(
#     repo_id="ernestumorga/ppo-Pendulum-v1",
#     filename="ppo-Pendulum-v1.zip"
# )
model = PPO.load('ppo-Pendulum-v1.zip')
# Evaluate the agent and watch it
eval_env = gym.make("Pendulum-v1")
mean_reward, std_reward = evaluate_policy(
    model, eval_env, render=False, n_eval_episodes=400, deterministic=True, warn=False
)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
