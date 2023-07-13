### https://stable-baselines.readthedocs.io/en/master/guide/examples.html
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

env = gym.make("CartPole-v1", n_envs=4)

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_cartpole")

del model
model = A2C.load("a2c_cartpole")

obs = A2C.load("a2c_cartpole")
while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
