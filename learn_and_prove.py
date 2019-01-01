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
parser.add_argument('--file',
                    help='filename of the trained weights')
args = parser.parse_args()

print(args.file)
print(args.train)
if args.file==None:
    filename='checkpoint.pth'
else:
    filename=args.file

env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
agent = Agent(state_size=37, action_size=4, seed=0)

if args.train:
    all_returns, avg_reward, best_avg_reward = dqn_interact(env, agent, filename=filename)
    # save training curves
    np.savez(filename+'.npz', all_returns=all_returns,
             avg_reward=avg_reward, best_avg_reward=best_avg_reward)

    # to continue ...
    print('Press [Q] on the plot window to continue ...') 
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(all_returns)), all_returns)
    plt.ylabel('Scores')
    plt.xlabel('Episode #')
    plt.show()
else:
    # load the weights from file
    agent.actor_local.load_state_dict(torch.load(filename))

# check performance of the agent
for i in range(10):
    state = reset(env, train_mode=False)
    score = 0
    for j in range(1000):
        action = agent.act(state)
        state, reward, done = step(env, action)
        score += reward
        if done:
            break
    print('Episode: {} || Score: {}'.format(i+1,score))

# close the environment
env.close()
