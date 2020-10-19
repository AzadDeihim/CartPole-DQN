from collections import deque
from torch import nn
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, hidden_size, input_size=4,
                 output_size=2):
        # input state, output q value for each output
        super(QNetwork, self).__init__()
        self.dense_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, X):
        out = self.dense_layers(X)
        return out

class ReplayMemory():
    def __init__(self, max_size=1000):
        #replay memeory is a deque
        self.memory = deque(maxlen=max_size)

    def addMemory(self, experience):
        '''
        adds memory

        :param experience: [previous state, action, reward, next_state]
        '''
        self.memory.append(experience)

    def empty(self):
        '''
        clear replay memory
        '''
        self.memory.clear()

    def sample(self, batch_size):
        '''
        get a batch from the replay memory

        :param batch_size:
        '''
        batch = np.random.choice(np.arange(len(self.memory)),
                               size=batch_size,
                               replace=False)
        return [self.memory[i] for i in batch]

    def populate(self, prepopulate_amount, env):
        '''
        make random actions until the replay memory has enough experiences

        :param prepopulate_amount: the smount of memories to populate
        :param env: the enviroment
        '''


        #reset evironment and take a random action to get cart moving
        env.reset()
        state, reward, done, _ = env.step(env.action_space.sample())
        state = np.reshape(state, [1, 4])


        for _ in range(prepopulate_amount):
            # make random action
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

            if done:
                #terminal state will be defined as [0,0,0,0] just as a placeholder
                next_state = np.zeros(state.shape)
                self.memory.append((state, action, reward, next_state))

                # reset evironment and take a random action to get cart moving
                env.reset()
                state, reward, done, _ = env.step(env.action_space.sample())
                state = np.reshape(state, [1, 4])
            else:
                self.memory.append((state, action, reward, next_state))
                state = next_state