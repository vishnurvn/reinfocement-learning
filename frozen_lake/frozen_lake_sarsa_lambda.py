import time
import random
from os import system
from collections import Counter

import gym
import numpy as np


env = gym.make('FrozenLake-v0')

state_size = 16
action_space = env.action_space.n
alpha = 0.1
gamma = 0.9
lambda_ = 0.9
state_action_vals = np.random.randn(state_size, action_space)
policy = np.zeros(state_size, dtype=int)
episodes = 20000
eps = 0.2


def select_action(policy, state, eps):
    sample = random.random()

    if sample > eps:
        return env.action_space.sample()
    else:
        return policy[state]


for ep in range(episodes):
    state = env.reset()
    action = select_action(policy, state, eps)
    done = False
    elig_trace = np.zeros((state_size, action_space))
    while not done:
        delta_vals = np.zeros((state_size, action_space))
        next_state, reward, done, _ = env.step(action)
        if done and reward == 0:
            reward = -1
        next_action = select_action(policy, state, eps)

        action_value = state_action_vals[state][action]
        next_action_value = state_action_vals[next_state][next_action]
        delta_vals[state][action] = reward + gamma * next_action_value - action_value
        elig_trace[state][action] = elig_trace[state][action] + 1

        state_action_vals = state_action_vals + alpha * delta_vals
        elig_trace = gamma * lambda_ * elig_trace

    policy = state_action_vals.argmax(axis=1)

print(policy)
system('cls')
state = env.reset()
done = False
while not done:
    action = policy[state]
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()
    time.sleep(1)
    system('cls') 
env.close()