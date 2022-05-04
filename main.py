import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Agent(object):

    def __init__(self, env, n_act, learning_rate, gamma):
        self.qshape = [item.n for item in env] + [n_act]
        self.n_act = n_act
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q = np.zeros(self.qshape)
        self.game_count = 0
        self.itcnt_Q = np.zeros(self.qshape)

    def epsilon_greedy(self, obs, epsilon, episode_id):
        x = np.random.uniform(0, 1)
        eps = max(0.05, epsilon - episode_id * 0.000003)
        print(episode_id, eps)
        if epsilon < 0:
            eps = 0
        if x < 1.0 - eps:
            j = (obs[0], obs[1], int(obs[2]))
            list_Q = self.Q[j]
            action = np.argmax(list_Q)
        else:
            action = np.random.choice(self.n_act)
        return action

    def get_sample(self, env, episode_id, epsilon):
        sample_obs, sample_reward, sample_action = [], [], []
        obs = env.reset()
        sample_obs.append(obs)
        is_done = False
        self.game_count = 0
        while not is_done:
            act = self.epsilon_greedy(obs, episode_id=episode_id, epsilon=epsilon)
            obs, reward, is_done, info = env.step(act)
            if not is_done:
                sample_obs.append(obs)
            sample_action.append(act)
            sample_reward.append(reward)
            self.game_count += 1
        return [sample_obs, sample_reward, sample_action]

    def update_Q(self, data):
        [sample_obs, sample_reward, sample_action] = data
        G = 0
        for i in range(len(sample_obs) - 1, -1, -1):
            G = G * self.gamma + sample_reward[i]
            if sample_obs[i] in sample_obs[:i] and sample_action[i] in sample_action[:i]:
                continue
            j = (sample_obs[i][0], sample_obs[i][1], sample_obs[i][2], sample_action[i])
            self.itcnt_Q[j] += 1
            self.Q[j] += (1 / (self.itcnt_Q[j])) * (G - self.Q[j])

    def train_process(self, env, iteration_times=50000):
        for episode_id in range(iteration_times):
            data = self.get_sample(env, episode_id=episode_id, epsilon=0.15)
            self.update_Q(data)

    def test_process(self, env, test_times=5000):
        win_count, draw_count, lose_count = 0, 0, 0
        for test_id in range(test_times):
            data = self.get_sample(env, epsilon=-1, episode_id=0)
            if data[1][-1] > 0:
                win_count += 1
            elif data[1][-1] == 0:
                draw_count += 1
            else:
                lose_count += 1
        print("\n==================TEST RESULT==================")
        print('win: ' + str(win_count) + ' lose: ' + str(lose_count) + ' draw: ' + str(draw_count))
        print('win_rate: ' + str(win_count / test_times) + ' lose_rate: ' + str(
            lose_count / test_times) + ' draw_rate: ' + str(draw_count / test_times))
        print("===============================================\n")

    def plot_figure(self):
        x = np.arange(0, 31, 1)
        y = np.arange(0, 10, 1)
        mx, my = np.meshgrid(x, y)

        def f(X, Y, c):
            return (self.Q[X, Y, c, 0] + self.Q[X, Y, c, 1]) / 2

        z0 = f(mx, my, 0)
        z1 = f(mx, my, 1)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(mx, my, z0)

        plt.show()


env = gym.make("Blackjack-v1", natural=True)
agent = Agent(
    env.observation_space,
    env.action_space.n,
    learning_rate=0.1,
    gamma=0.1
)

agent.train_process(env, iteration_times=50000)
agent.test_process(env, test_times=5000)
agent.plot_figure()
