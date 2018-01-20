# coding:utf-8
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf


class QNet:
    def __init__(self, learning_rate=0.01, learning_rate_decay=0.01, state_size=4, action_size=2):
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=state_size, activation='relu'))
        self.model.add(Dense(48, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=learning_rate, decay=learning_rate_decay))