import time
import pickle
import os
import gym
import numpy as np

env = gym.make("FrozenLake-v0")

epsilon = 0.9
total_episodes = 10000
max_steps = 100

learning_rate = 0.0
gamma = 0.96

q_table = np.zeros((env.observation_space.n, env.action_space.n))


def choose_action(state):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state, :])
    return action


def update(state, action, reward, state1):
    pred = q_table[state, action]
    target = reward + gamma * np.max(q_table[state1, :])
    q_table[state, action] = q_table[state, action] + learning_rate * (
        target - pred
    )


for episode in range(total_episodes):
    state = env.reset()
    t = 0
    print(f"episode {episode}...")

    while t < max_steps:
        # env.render()
        action = choose_action(state)
        state1, reward, done, _ = env.step(action)
        update(state, action, reward, state1)

        state = state1
        t += 1

        if done:
            break

        time.sleep(0.1)

print(q_table)

with open("./frozen_lake-q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)
