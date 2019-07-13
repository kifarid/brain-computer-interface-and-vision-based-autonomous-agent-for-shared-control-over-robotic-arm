import numpy as np
import gym
env = gym.make("FetchPickAndPlace-v1")
observation = env.reset()
print(np.unique(observation['image']))
for _ in range(1000):
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)


  if done:
    observation = env.reset()
env.close()
