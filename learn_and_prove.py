from dqn_agent import Agent
from dqn_monitor import dqn_interact, reset, step
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true',
                    help='run a pre-trainded neural network agent')
parser.add_argument('--random', action='store_true',
                    help='run a tabula rasa agent')
parser.add_argument('--display', action='store_true',
                    help='display scores per episode')
parser.add_argument('--file',
                    help='filename of the trained weights')
args = parser.parse_args()

# setting filename
if args.file==None:
    filename='checkpoint.pth'
else:
    filename=args.file
# current configuration
config = {
    'n_episodes':       400,  # max. number of episode to train the agent
    'window':           100,  # save the last XXX returns of the agent
    'max_t':            500,  # max. number of steps per episode
    'eps_start':        1.0,  # GLIE parameters
    'eps_end':        0.005,  #
    'eps_decay':      0.960,  #
    'BUFFER_SIZE': int(1e5),  # replay buffer size
    'BATCH_SIZE':        16,  # minibatch size
    'GAMMA':           0.99,  # discount factor
    'TAU':             1e-3,  # for soft update or target parameters
    'LR':              5e-4,  # learning rate
    'UPDATE_EVERY':       1,  # how often to update the network
    'FC1_UNITS':         16,  # number of neurons in fisrt layer
    'FC2_UNITS':         16,  # number of neurons in second layer
    }
# print configuration
print(' Config Parameters')
for k,v in config.items():
    print('{:<15}: {:>15}'.format(k,v))
#print file namespace
print('File namespace: {}'.format(filename))

env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
agent = Agent(state_size=37, action_size=4, seed=0,
              fc1_units=config['FC1_UNITS'],
              fc2_units=config['FC2_UNITS'],
              buffer_size=config['BUFFER_SIZE'],
              batch_size=config['BATCH_SIZE'],
              update_every=config['UPDATE_EVERY'],
              learning_rate=config['LR'],
              tau=config['TAU'],
              gamma=config['GAMMA'])

# print neural network architecture
print(agent.actor_local)

# to prove policy
eps = 0.0
if args.train:
    # begin training interaction
    all_returns, avg_reward, best_avg_reward = dqn_interact(env, agent, filename=filename,
                                                            n_episodes=config['n_episodes'],
                                                            window=config['window'],
                                                            max_t=config['max_t'],
                                                            eps_start=config['eps_start'],
                                                            eps_end=config['eps_end'],
                                                            eps_decay=config['eps_decay'])
    # save training curves
    np.savez(filename+'.npz', all_returns=all_returns,
             avg_reward=avg_reward, best_avg_reward=best_avg_reward)
    # display mode
    if args.display:
        # to continue ...
        print('Press [Q] on the plot window to continue ...')
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(all_returns)), all_returns)
        plt.ylabel('Scores')
        plt.xlabel('Episode #')
        plt.show()
elif args.random:
    # choose random policy
    eps = 1.0
else:
    # load the weights from file
    agent.actor_local.load_state_dict(torch.load(filename))

# check performance of the agent
for i in range(10):
    state = reset(env, train_mode=False)
    score = 0
    for j in range(1000):
        action = agent.act(state, eps=eps)
        state, reward, done = step(env, action)
        score += reward
        if done:
            break
    print('Episode: {} || Score: {}'.format(i+1,score))

# close the environment
env.close()
