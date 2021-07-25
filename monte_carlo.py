import gym
import numpy as np
from functools import reduce
from RL.marcov_decision_process_with_rewards import value_iteration

gamma = 0.95
EPS = 0.00000001
ITERATION_COUNT = 20000
SLIPERTY = True
OPTIMAL = True
PRINT = True
env = gym.make('FrozenLake-v0', is_slippery=SLIPERTY)
env.render()

nrof_action = env.action_space.n         # env.unwrapped.nA # количество действий
nrof_states = env.observation_space.n    # env.unwrapped.nS # кол-во сост
print(f"action count: {nrof_action}\nstate count: {nrof_states}")


if SLIPERTY:
    optimal_policy = np.array(
        [[1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
         ]
    )
else:
    optimal_policy = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

def first_visit_policy_evaluation():
    V = np.zeros(nrof_states)
    if OPTIMAL:
        policy = optimal_policy
    else:
        policy = np.random.rand(nrof_states, nrof_action)
        policy = policy / policy.sum(axis=1).reshape(-1, 1)

    returns = [[] for _ in range(nrof_states)]

    iteration = 0
    visit = np.zeros(nrof_states)

    actions = np.arange(nrof_action)
    while iteration < ITERATION_COUNT:
        iteration += 1
        print(f"\riteration {iteration}", end='')
        state = env.reset()
        visit *= 0
        done = False
        game_path_state_count = 0
        reward = 0
        while not done:
            action = np.random.choice(actions, p=policy[state, :])
            state, reward, done, _ = env.step(action)
            game_path_state_count += 1
            if visit[state] == 0:
                visit[state] = game_path_state_count

        for _state in range(nrof_states):
            if visit[_state] != 0: # last reward
                returns[_state].append( reward * gamma ** (game_path_state_count - visit[_state]) )


    print()
    for state in range(nrof_states):
        if returns[state]:
            V[state] = sum(returns[state]) / len(returns[state])

    print("\nfirst_visit_policy_evaluation/values_field:")
    for i in range(0, nrof_states, 4):
        print(np.round(V[i:i + 4],2))

    return V


def every_visit_policy_evaluation():
    V = np.zeros(nrof_states)
    if OPTIMAL:
        policy = optimal_policy
    else:
        policy = np.random.rand(nrof_states, nrof_action)
        policy = policy / policy.sum(axis=1).reshape(-1, 1)

    returns = [[] for _ in range(nrof_states)]
    iteration = 0


    actions = np.arange(nrof_action)
    while iteration < ITERATION_COUNT:
        iteration += 1
        print(f"\riteration {iteration}", end='')
        state = env.reset()
        visits = [[] for _ in range(nrof_states)]
        done = False
        game_step_count = 0
        reward = 0
        while not done:
            action = np.random.choice(actions, p=policy[state, :])
            state, reward, done, _ = env.step(action)
            game_step_count += 1
            visits[state].append(game_step_count)

        for state in range(nrof_states):
            if visits[state]:  # last reward
                for step in visits[state]:
                    returns[state].append(reward * (gamma ** (game_step_count - step)))

    print()
    for state in range(nrof_states):
        if returns[state]:
            V[state] = sum(returns[state]) / len(returns[state])

    print("\nevery_visit_policy_evaluation/values_field:")
    for i in range(0, nrof_states, 4):
        print(np.round(V[i:i + 4],2))

    return V




def MC_onPolicy_control():
    EPS = 0.1
    ITERATION_COUNT = 13000
    main_action_prob, another_prob = 1 - EPS + EPS/nrof_action, EPS/nrof_action
    policy = np.ones((nrof_states, nrof_action)) * another_prob
    policy[np.arange(nrof_states), np.random.randint(0, 4, size=nrof_states)] = 1 - EPS + EPS/nrof_action

    iteration = 0
    Q = np.zeros((nrof_states, nrof_action))
    count = np.zeros((nrof_states, nrof_action), dtype=np.uint32)
    visited = np.zeros((nrof_states, nrof_action))

    actions = np.arange(nrof_action)
    states = np.arange(nrof_states)

    while iteration < ITERATION_COUNT:

        iteration += 1
        print(f"\riteration {iteration}/{ITERATION_COUNT}", end='')
        visited *= 0
        state = env.reset()
        done = False
        game_path_state_count, reward, action = 0, 0, 0
        while not done:
            game_path_state_count += 1
            action = np.random.choice(actions, p=policy[state, :])
            if visited[state, action] == 0:
                visited[state, action] = game_path_state_count
            state, reward, done, _ = env.step(action)

        Q[visited != 0] = (Q[visited != 0] * count[visited != 0] + reward*(gamma**(game_path_state_count - visited[visited != 0]))) / (count[visited != 0] + 1)
        count[visited != 0] += 1

        m_more0_ind = np.where(np.max(Q, axis=1) > 0)[0]
        policy[states[m_more0_ind], np.argmax(policy, axis=1)[m_more0_ind]] = another_prob
        policy[states[m_more0_ind], np.argmax(Q, axis=1)[m_more0_ind]] = main_action_prob

    if PRINT:
        print()
        op = np.argmax(Q, axis=1)
        Q = np.round(Q, 2)

        print(count)
        print("OnPolicy/Q_matrix:")
        print(Q)

        print("Optimal_policy:")
        print(op)
        for i in range(0,nrof_states,4):
            print(op[i:i+4])


def MC_offPolicy_control():
    EPS = 0.1
    ITERATION_COUNT = 100000
    main_action_prob, another_prob = 1 - EPS + EPS / nrof_action, EPS / nrof_action
    mu_policy = np.ones((nrof_states, nrof_action))
    policy = np.random.randint(0, 4, size=nrof_states)

    iteration = 0
    Q = np.zeros((nrof_states, nrof_action))
    C = np.zeros((nrof_states, nrof_action))
    actions = np.arange(nrof_action)

    while iteration < ITERATION_COUNT:
        mu_policy[:, :] = another_prob
        mu_policy[np.arange(nrof_states), np.random.randint(0, 4, size=nrof_states)] = main_action_prob

        iteration += 1
        print(f"\riteration {iteration}/{ITERATION_COUNT}", end='')

        state = env.reset()
        done = False
        g,w = 0, 1
        episode = []
        while not done:
            action = np.random.choice(actions, p=mu_policy[state, :])
            new_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = new_state

        for s,a,r in reversed(episode):
            g = gamma*g + r
            C[s,a] = C[s,a] + w
            Q[s,a] = Q[s,a] + w*(g-Q[s,a])/C[s,a]
            policy[s] = np.argmax(Q[s,:])
            w = w / mu_policy[s,a]
            if a != policy[s]: break

    if PRINT:
        print()
        op = np.argmax(Q, axis=1)
        Q = np.round(Q, 2)

        print("OnPolicy/Q_matrix:")
        print(Q)

        print("Optimal_policy:")
        print(op)
        for i in range(0,nrof_states, 4):
            print(op[i:i+4])






if __name__ == "__main__":
    if 0:
        V_emp = first_visit_policy_evaluation()
        V = value_iteration()
        print(f"\n BIAS: {np.mean(V - V_emp)} VAR: {np.var(V-V_emp)}")
        V_emp = every_visit_policy_evaluation()
        print(f"\n BIAS: {np.mean(V - V_emp)} VAR: {np.var(V-V_emp)}")
    else:
        MC_onPolicy_control()
        #MC_offPolicy_control()
