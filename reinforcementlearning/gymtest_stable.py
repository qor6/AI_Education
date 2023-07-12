import gymnasium as gym
import cv2
from stable_baselines3 import PPO

# env = gym.make("CartPole-v1", render_mode='rgb_array')
# env = gym.make("LunarLander-v2", render_mode='rgb_array')   #pip install gymnasium[box2d] swig
# env = gym.make("MountainCar-v0", render_mode='rgb_array')
# env = gym.make("Pendulum-v1", render_mode='rgb_array')
env = gym.make("Ant-v4", render_mode='rgb_array')

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

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
