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
state_action_vals = np.random.randn(state_size, action_space)
state_action_vals[-1] = 0
policy = np.zeros(state_size, dtype=int)
episodes = 10000
EPS_END = 0.05
EPS_START = 0.99
EPS_DECAY = 200
eps = EPS_START
counter = Counter()


def select_action(policy, state, eps):
    eps = EPS_END * (1 - np.exp(-1/EPS_DECAY)) + eps * np.exp(-1/EPS_DECAY)
    sample = random.random()

    if sample > eps:
        return policy[state], eps
    else:
        return env.action_space.sample(), eps


for ep in range(episodes):
    state = env.reset()
    action, eps = select_action(policy, state, eps)
    done = False
    visited = []
    while not done:
        next_state, reward, done, _ = env.step(action)
        next_action, eps = select_action(policy, next_state, eps)
        delta = reward + gamma * state_action_vals[next_state, next_action] - state_action_vals[state, action]
        state_action_vals[state, action] = state_action_vals[state, action] + alpha * delta
        state, action = next_state, next_action
        visited.append(state)

    policy = state_action_vals.argmax(axis=1)
    counter.update(visited)

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
