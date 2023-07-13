### https://stable-baselines.readthedocs.io/en/master/guide/examples.html
import gym

# from stable_baselines3.common.policies import MlpPolicy
# from stable_baselines3.common import make_vec_env
# from stable_baselines3 import A2C

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.dqn.policies import MlpPolicy, CnnPolicy
# from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN


env = make_atari_env("BreakoutNoFrameskip-v4")

model = DQN(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("DQN_Breakout")

del model
model = DQN.load("DQN_Breakout")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render(mode="human")
