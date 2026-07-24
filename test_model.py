import gymnasium as gym
import mani_skill.envs
import envs.reacher
import torch
from ppo import Agent

env = gym.make(
    "ReacherApto-v1",
    num_envs=1,
    obs_mode="state",
    control_mode="pd_joint_delta_pos_left_arm",
    render_mode="human"
)

model = Agent(env)
model.load_state_dict(torch.load('runs/ReacherApto-v1__ppo__1__1784860804/final_ckpt.pt'))

obs, _ = env.reset(seed=0)
for _ in range(1000):
    action = model.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
    env.render()
env.close()