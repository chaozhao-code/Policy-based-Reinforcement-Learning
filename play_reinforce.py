from ActorCritic import ACAgent
from reinforce import REINFORCEAgent
from catch import Catch
import matplotlib
import tqdm
matplotlib.use('TkAgg') #'Qt5Agg') # 'TkAgg'
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def test():
    # Hyperparameters
    rows = 7
    columns = 7
    speed = 2
    max_steps = 250
    max_misses = 10
    observation_type = 'vector'  # 'vector'
    seed = None

    episodes=10000

    # Initialize environment and Q-array
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)
    step_pause = 0.3  # the pause between each plot
    # env.render(step_pause)

    n_states = 3
    n_actions = env.action_space.n

    agent = ACAgent(env, n_states, n_actions, learning_rate=0.001, lamda=0.5, step=10, bootstrapping=True, baseline=True, ClipPPO=True)
    # agent = REINFORCEAgent(env, n_states, n_actions, ClipPPO=True)
    #train
    for i in range(5000):
        print("Episode: ", i, "Reward: ", agent.train(i))

    #play
    s = env.reset()
    terminal = False
    while not terminal:
        a, _, _ = agent.actor(s)
        next_s, reward, terminal, _ = env.step(a)
        s = next_s
        env.render(step_pause)

if __name__ == '__main__':
    test()