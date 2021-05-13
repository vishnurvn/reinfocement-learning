import gym
import numpy as np

environment = gym.make('MountainCar-v0')

ALPHA = 1
GAMMA = 0.9
EPSILON = 0.8


def get_polynomial_encoding(state):
    s1, s2 = state
    return np.array([1, s1, s2, s1*s2])


class SARSA:
    def __init__(self, env, weights, epsilon, test=False):
        self.env = env
        self.weights = weights
        self.episodes = []
        self.rewards = []
        self.test = test
        self.epsilon = epsilon
        self.current_eps = self.epsilon

    def get_action(self, state):
        return np.dot(self.weights.T, state).argmax()

    def get_epsilon_greedy_action(self, state):
        sample = np.random.uniform()
        if sample > self.current_eps:
            return self.get_action(state)
        else:
            return self.env.action_space.sample()

    def decay_epsilon(self, ep, n_ep):
        # self.current_eps = self.epsilon * 1 / (1 + np.exp(16 * ep / n_ep - 9))
        self.current_eps = self.epsilon

    def test_agent(self, n_episodes):
        for _ in n_episodes:
            done = False
            state = self.env.reset()
            while not done:
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
            self.env.close()

    def compute_gradient(self, state, action, next_state, next_action, reward):
        next_state_action_val = np.dot(self.weights.T, next_state)[next_action]
        state_action_val = np.dot(self.weights.T, state)[action]
        grad = (reward + GAMMA * next_state_action_val - state_action_val) * state
        return grad

    def optimize_agent(self, n_episodes):
        for ep in range(n_episodes):
            state = self.env.reset()
            state = get_polynomial_encoding(state)
            done = False
            ep_reward = 0
            while not done:
                action = self.get_epsilon_greedy_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = get_polynomial_encoding(next_state)
                next_action = self.get_epsilon_greedy_action(next_state)
                grad = self.compute_gradient(state, action,
                                             next_state, next_action, reward)
                self.weights[:, action] += ALPHA * grad
                state, action = next_state, next_action
                ep_reward += reward
            self.episodes.append(ep)
            self.rewards.append(ep_reward)
            self.decay_epsilon(ep, n_episodes)

    def play_agent(self):
        done = False
        state = get_polynomial_encoding(self.env.reset())
        while not done:
            action = self.get_action(state)
            state, reward, done, _ = self.env.step(action)
            state = get_polynomial_encoding(state)
            self.env.render()
        self.env.close()


if __name__ == '__main__':
    action_space = environment.action_space.n
    obs_space = environment.observation_space.shape[0] + 2
    weights = np.random.randn(obs_space, action_space)
    sarsa = SARSA(environment, weights.copy(), EPSILON)
    sarsa.optimize_agent(1000)
    sarsa.play_agent()
    print(weights)
    print(sarsa.weights)
