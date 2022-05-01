import gym
import numpy as np


class Agent(object):

    def __init__(self, n_obs, n_act, learning_rate, gamma):
        self.n_obs = n_obs
        self.n_act = n_act
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q = np.zeros((n_obs, n_act))
        self.game_count = 0

    def clear(self):
        self.game_count = 0

    def epsilon_greedy(self, env, epsilon=0.1):
        x = np.random.uniform(0, 1)
        if x < 1.0 - epsilon:
            list_Q = self.Q[env, :]
            max_Q = np.max(list_Q)
            candidate_action = np.where(list_Q == max_Q)[0]
            action = np.random.choice(candidate_action)
        else:
            action = np.random.choice(self.n_act)
        return action

    def update_Q(self, env, action, reward, next_env, is_done):
        predict = self.Q[env, action]


env = gym.make("Blackjack-v1", natural=True)
agent = Agent(
    env.observation_space.n,
    env.action_space.n,
    learning_rate=0.1,
    gamma=0.9
)


def train_agent():


def test_agent():


train_agent()
test_agent()
