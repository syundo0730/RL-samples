# coding:utf-8
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import q

class DQNCartPoleSolver():
    def __init__(self, n_episodes=3000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v0')
        if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.quiet = quiet
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Q Models
        self.mainQ = q.QNet(learning_rate=alpha, learning_rate_decay=alpha_decay)
        self.subQ = q.QNet(learning_rate=alpha, learning_rate_decay=alpha_decay)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, episode):
        epsilon = 0.001 + 0.9 / (1.0+episode)
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.mainQ.model.predict(state))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self._getDQNTarget(state, action, reward, next_state, done)
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self.mainQ.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _getDQNTarget(self, state, action, reward, next_state, done):
        target = self.mainQ.model.predict(state)
        target[0][action] = reward if done else reward + self.gamma * np.max(self.subQ.model.predict(next_state)[0])
        return target

    def _getDDQNTarget(self, state, action, reward, next_state, done):
        target = self.mainQ.model.predict(state)
        next_action = np.argmax(target[0])
        target[0][action] = reward if done else reward + self.gamma * self.subQ.model.predict(next_state)[0][next_action]
        return target

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0

            if e % 2 == 0:
                self.mainQ = self.subQ

            while not done:
                action = self.choose_action(state, e)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                return e - 100
            if e % 100 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

            self.replay(self.batch_size)

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()