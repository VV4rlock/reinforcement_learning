import gym
import numpy as np
from functools import reduce
from RL.marcov_decision_process_with_rewards import value_iteration

gamma = 0.95
SLIPERTY = False

PRINT = True
env = gym.make('FrozenLake-v0', is_slippery=SLIPERTY)
env.render()

nrof_action = env.action_space.n         # env.unwrapped.nA # количество действий
nrof_states = env.observation_space.n    # env.unwrapped.nS # кол-во сост
#print(f"action count: {nrof_action}\nstate count: {nrof_states}")

def SARSA():
    EPS = 0.1
    ITERATION_COUNT = 3000
    alpha = 0.9
    main_action_prob, another_prob = 1 - EPS + EPS/nrof_action, EPS/nrof_action
    policy = np.ones((nrof_states, nrof_action)) * another_prob
    policy[np.arange(nrof_states), np.random.randint(0, 4, size=nrof_states)] = 1 - EPS + EPS/nrof_action

    iteration = 0
    Q = np.random.uniform(0, 1, (nrof_states, nrof_action))
    visited = np.zeros((nrof_states, nrof_action))

    actions = np.arange(nrof_action)

    while iteration < ITERATION_COUNT:

        iteration += 1
        print(f"\riteration {iteration}/{ITERATION_COUNT}", end='')
        visited *= 0
        state = env.reset()
        action = np.random.choice(actions, p=policy[state, :])
        reward, done = 0, False
        while not done:
            prev_action, prev_state = action, state
            state, reward, done, _ = env.step(action)
            action = np.random.choice(actions, p=policy[state, :])

            if done:
                Q[state, :] = 0

            Q[prev_state, prev_action] = Q[prev_state, prev_action] + alpha * (reward + gamma*Q[state, action] - Q[prev_state, prev_action])

            policy[prev_state, :] = another_prob
            policy[prev_state, np.argmax(Q[prev_state, :])] = main_action_prob

    if PRINT:
        print()
        op = np.argmax(Q, axis=1)
        Q = np.round(Q, 2)

        print("SARSA/Q_matrix:")
        print(Q)

        print("SARSA/Optimal_policy:")
        print(op)
        for i in range(0,nrof_states,4):
            print(op[i:i+4])


def Q_learning():
    EPS = 0.1
    ITERATION_COUNT = 1000
    alpha = 0.9
    main_action_prob, another_prob = 1 - EPS + EPS/nrof_action, EPS/nrof_action
    policy = np.ones((nrof_states, nrof_action)) * another_prob
    policy[np.arange(nrof_states), np.random.randint(0, 4, size=nrof_states)] = 1 - EPS + EPS/nrof_action

    iteration = 0
    Q = np.random.uniform(0, 1, (nrof_states, nrof_action))
    visited = np.zeros((nrof_states, nrof_action))

    actions = np.arange(nrof_action)

    while iteration < ITERATION_COUNT:

        iteration += 1
        print(f"\riteration {iteration}/{ITERATION_COUNT}", end='')
        visited *= 0
        state = env.reset()
        reward, done = 0, False
        while not done:
            prev_state = state
            action = np.random.choice(actions, p=policy[state, :])
            state, reward, done, _ = env.step(action)

            if done:
                Q[state, :] = 0

            Q[prev_state, action] = Q[prev_state, action] + alpha * (reward + gamma*np.max(Q[state, :]) - Q[prev_state, action])

            policy[prev_state, :] = another_prob
            policy[prev_state, np.argmax(Q[prev_state, :])] = main_action_prob

    if PRINT:
        print()
        op = np.argmax(Q, axis=1)
        Q = np.round(Q, 2)

        print("Q_learning/Q_matrix:")
        print(Q)

        print("Q_learning/Optimal_policy:")
        print(op)
        for i in range(0,nrof_states,4):
            print(op[i:i+4])


def double_Q_learning():
    EPS = 0.1
    ITERATION_COUNT = 1000
    alpha = 0.9
    main_action_prob, another_prob = 1 - EPS + EPS/nrof_action, EPS/nrof_action
    policy = np.ones((nrof_states, nrof_action)) * another_prob
    policy[np.arange(nrof_states), np.random.randint(0, 4, size=nrof_states)] = 1 - EPS + EPS/nrof_action

    iteration = 0
    Q1 = np.random.uniform(0, 1, (nrof_states, nrof_action))
    Q2 = np.random.uniform(0, 1, (nrof_states, nrof_action))
    visited = np.zeros((nrof_states, nrof_action))

    actions = np.arange(nrof_action)

    while iteration < ITERATION_COUNT:

        iteration += 1
        print(f"\riteration {iteration}/{ITERATION_COUNT}", end='')
        visited *= 0
        state = env.reset()
        reward, done = 0, False
        while not done:
            prev_state = state
            action = np.random.choice(actions, p=policy[state, :])
            state, reward, done, _ = env.step(action)

            if done:
                Q1[state, :] = 0
                Q2[state, :] = 0 # так проще) неважно что на первых итерация терминальное состояние имеет не 0 Q

            if np.random.rand() > 0.5:
                Q1[prev_state, action] = Q1[prev_state, action] + alpha * (reward + gamma * Q2[state, np.argmax(Q1[state, :])] - Q1[prev_state, action])
            else:
                Q2[prev_state, action] = Q2[prev_state, action] + alpha * (
                            reward + gamma * Q1[state, np.argmax(Q2[state, :])] - Q2[prev_state, action])


            policy[prev_state, :] = another_prob
            policy[prev_state, np.argmax(Q1[prev_state, :] + Q2[prev_state, :])] = main_action_prob

    if PRINT:
        print()
        op = np.argmax(Q1 + Q2, axis=1)
        Q = np.round(Q1 + Q2, 2)

        print("double_Q_learning/Q_matrix:")
        print(Q)

        print("double_Q_learning/Optimal_policy:")
        print(op)
        for i in range(0,nrof_states,4):
            print(op[i:i+4])

if __name__=="__main__":
    SARSA()
    #Q_learning()
    #double_Q_learning()
