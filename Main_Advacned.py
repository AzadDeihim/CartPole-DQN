import gym
from collections import deque
from torch import nn
import torch
import copy
from torch.utils import data
import numpy as np
import random
from torchvision import models
from hyperopt import Trials, fmin, STATUS_OK, tpe, hp
import matplotlib.pyplot as plt
from Agent_Advacned import QNetwork, ReplayMemory

# make cartpole environment
env = gym.make('CartPole-v0')

memory_size = 10000
memory = ReplayMemory(max_size=memory_size)

eval = []


def objective(hyperparameters):
    dict = {'loss': 1000,
            'status': STATUS_OK
            }
    episodes = 1000
    max_steps = 151

    #define parameters
    gamma = hyperparameters['gamma']
    hidden_size = int(hyperparameters['hidden_size'])
    learning_rate = hyperparameters['learning_rate']
    epsilon = hyperparameters['epsilon']
    batch_size = 32
    pretrain_length = batch_size


    # run on gpu if possible
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = QNetwork(hidden_size=hidden_size)
    model.train()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()





    all_rewards = []
    memory.populate(pretrain_length, env)
    step = 0
    env.reset()

    #make random action to get cart moving
    state, reward, done, _ = env.step(env.action_space.sample())
    state = np.reshape(state, [1, 4])

    for ep in range(1, episodes):

        total_reward = 0
        t = 0

        #training phase
        while t < max_steps:
            model.eval()
            step += 1

            #explore or exploit
            if epsilon > np.random.rand():
                action = env.action_space.sample()
            else:
                state = torch.tensor(state)
                state = state.type('torch.FloatTensor').to(device)
                Qs = model(state)
                _, action = torch.max(Qs, 1)
                action = action.item()

            # make action, get reward and new state
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            total_reward += reward


            #if new state is terminal state
            if done:
                #terminal state will be defined as [0,0,0,0] just as a placeholder
                next_state = np.zeros(state.shape)

                #set t to max steps so while loop will terminate
                t = max_steps

                #add experiance to replay memeory
                memory.addMemory((state, action, reward, next_state))

                #reset env and make random action to get the cart moving
                env.reset()
                state, reward, done, _ = env.step(env.action_space.sample())
                state = np.reshape(state, [1, 4])
            else:
                memory.addMemory((state, action, reward, next_state))
                state = next_state
                t += 1

            # Replay
            inputs = np.zeros((batch_size, 4))
            targets = np.zeros((batch_size, 2))

            #get a batch from the replay memeory to train the network
            minibatch = memory.sample(batch_size)

            for i, (state_batch, action_batch, reward_batch, next_state_batch) in enumerate(minibatch):
                inputs[i:i+1] = np.array(state_batch)
                target = reward_batch

                #if not terminal state
                if not (next_state_batch == np.zeros(state_batch.shape)).all(axis=1):

                    #calculate target for network
                    next_state_b = torch.tensor(next_state_batch)
                    next_state_b = next_state_b.type('torch.FloatTensor').to(device)
                    target_Q = model(next_state_b)
                    _, dontcare = torch.max(target_Q, 1)
                    target = reward_batch + gamma * _.item()


                state_b = torch.tensor(state_batch)
                state_b = state_b.type('torch.FloatTensor').to(device)
                targets[i] = model(state_b).detach().numpy().reshape(2,)
                if torch.is_tensor(action_batch):
                    targets[i][action_batch.item()] = target
                else:
                    targets[i][action_batch] = target


            #train network using batch from replay memory
            model.train()
            for i in range(len(inputs)):
                input = torch.tensor(inputs[i])
                target = torch.tensor(targets[i])
                input = input.type('torch.FloatTensor').to(device)
                target = target.type('torch.FloatTensor').to(device)
                optimizer.zero_grad()
                out = model(input)
                loss = loss_function(out, target)
                loss.backward()
                optimizer.step()

        #decay epsilon
        if epsilon >= 0.5:
            epsilon *= 0.99999
        else:
            epsilon *= 0.9999






        t = 0
        total_reward = 0
        #testing phase
        while t < max_steps:
            model.eval()

            #exploit only
            state = torch.tensor(state)
            state = state.type('torch.FloatTensor').to(device)
            Qs = model(state)
            _, action = torch.max(Qs, 1)
            action = action.item()

            # make action, get reward and new state
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            total_reward += reward

            #if terminal state
            if done:
                next_state = np.zeros(state.shape)
                all_rewards.append(total_reward)
                t = max_steps
                env.reset()
                state, reward, done, _ = env.step(env.action_space.sample())
                state = np.reshape(state, [1, 4])
            else:
                state = next_state
                t += 1


    # 150 - because the bayesian search will minimize
    dict['loss'] = 150 - np.max(all_rewards)
    print('High Score: ' + str(np.max(all_rewards)))
    print('Epsilon: ' + str(hyperparameters['epsilon']))
    print('Gamma: ' + str(hyperparameters['gamma']))
    print('Learning Rate: ' + str(hyperparameters['learning_rate']))
    print('Hidden Size: ' + str(hyperparameters['hidden_size']))

    eval.append([hyperparameters['epsilon'], hyperparameters['learning_rate'], hyperparameters['gamma'],
                                              hyperparameters['hidden_size'], dict['loss'], np.argmax(all_rewards)])
    np.save('bayes_search_advanced_3', eval)
    return dict


#define search space for parameters
space = {
    'epsilon': hp.uniform('epsilon', 0, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
    'gamma': hp.uniform('gamma', 0, 1),
    'hidden_size': hp.quniform('hidden_size', 2, 100, q=2),
}

#set bayesian search to run for 100 iterations
bayes_trials = Trials()
MAX_EVALS = 100
# Optimise
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS)
