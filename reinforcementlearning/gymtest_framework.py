import gymnasium as gym
import cv2
from stable_baselines3 import PPO
####environment list :https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/__init__.py
### theory after : https://gymnasium.farama.org/environments/box2d/lunar_lander/

env = gym.make("LunarLander-v2", render_mode='rgb_array')   #pip install gymnasium[box2d], swig, pip3 install box2d box2d-kengz


model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_00)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render(mode="human")
    # VecEnv resets automatically
    # if done:
    #     obs = env.reset()

env.close()
