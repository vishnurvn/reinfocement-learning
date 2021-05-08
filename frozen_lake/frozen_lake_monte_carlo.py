import gym
import time
from os import system

import numpy as np


env = gym.make('FrozenLake-v0')

state_size = 16
action_space = env.action_space.n
alpha = 0.1
gamma = 0.9
state_action_vals = np.zeros((state_size, action_space))
policy_dict = np.zeros(state_size, dtype=int)
episodes = 100000


for ep in range(episodes):
    state = env.reset()
    done = False
    value_return = 0
    time_step = {}
    ep_time_step = 0
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        if done and reward == 0:
            reward = -1
        time_step[ep_time_step] = {
            'state': state,
            'reward': reward,
            'action': action
        }
        ep_time_step += 1
        state = next_state

    for step, d in time_step.items():
        value_return = value_return + gamma * reward
        state = d['state']
        action = d['action']

        action_value = state_action_vals[state][action]
        state_action_vals[state][action] = action_value + alpha * (value_return - action_value)
        state_actions = state_action_vals[state]
        policy_dict[state] = state_actions.argmax()

    if ep % 10000 == 0:
        print(f"Episode: {ep}, value_return: {value_return}")
        

state = env.reset()
done = False
while not done:
    action = policy_dict[state]
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()
    time.sleep(1)
    system('cls') 

env.close()