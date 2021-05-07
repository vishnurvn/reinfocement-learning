import gym
import time


env = gym.make('FrozenLake-v0')

alpha = 0.1
gamma = 0.9
state_dict = {}
policy_dict = {}
episodes = 1000


for ep in range(episodes):
    state = env.reset()
    done = False
    value_return = 0
    time_step = {}
    ep_time_step = 0
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
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

        if state not in state_dict:
            state_dict[state] = {
                0: 0,
                1: 0,
                2: 0,
                3: 0
            }
        action_value = state_dict[state][action]
        state_dict[state][action] = action_value + alpha * (value_return - action_value)
        state_actions = state_dict[state]
        max_action = max(state_actions.items(), key=lambda x: x[1])[0]
        policy_dict[state] = max_action


state = env.reset()
done = False
while not done:
    action = policy_dict[state]
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()
    time.sleep(1)
env.close()