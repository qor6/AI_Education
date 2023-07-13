# https://gymnasium.farama.org/environments/atari/complete_list/
import gym

# from stable_baselines3.common.policies import MlpPolicy
# from stable_baselines3.common import make_vec_env
# from stable_baselines3 import A2C

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.dqn.policies import MlpPolicy, CnnPolicy
from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3 import DQN
from stable_baselines3 import A2C


env = make_atari_env("BreakoutNoFrameskip-v4")
env = VecFrameStack(env, n_stack=4)

model = A2C("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("deepd_Breakout")

del model
model = A2C.load("deepd_Breakout")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render(mode="human")
