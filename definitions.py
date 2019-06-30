import numpy as np
import collections as cl
from copy import deepcopy as dpc
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib.eager as tfe
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Sequential  # linear stack of layers
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, AveragePooling1D
from tensorflow.keras.optimizers import Adam

'''
def update_policy( policy, optimizer ):
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    # if len(rewards) > 1 : rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []
'''
'''
def getGPUs():
    devices = device_lib.list_local_devices()
    return [ device.name for device in devices if device.device_type == 'GPU']'''

# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = cl.deque(maxlen=20000000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))

        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    def remember(self, state, action, reward, next_state, done):
        for s in range(len(state)):
            self.memory.append((state[s], action[s], reward[s], next_state[s], done[s]))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state=np.reshape(state,[1,len(state)])
            next_state=np.reshape(next_state,[1,len(next_state)])
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=100, verbose=2)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def saveModel( self, name ) :
        self.model.save('models/' + name + '.h5')
