import gym
import numpy as np


gamma = 0.95
EPS = 0.00000001
env = gym.make('FrozenLake-v0', is_slippery=False)
env.render()

PRINT =False
nrof_action = env.unwrapped.nA # количество действий
nrof_states = env.unwrapped.nS # кол-во сост
print(f"action count: {nrof_action}\nsate count: {nrof_states}")

'''
0 - left, 1 -down , 2 - right, 3 - up
env.unwrapped.P = env.unwrapped.P = {0: # Индекс состояния:
                                        {0: # Индекс действия (0 - left):
                                            [ # Cписок достижимых состояний:
                                            (1.0, # (Вероятность перехода,
                                            0, # индекс состояния,
                                            0.0, # награда,
                                            False) # является ли состояние терминальным)
                                            ],
                                        }
}
'''
#print(env.unwrapped.P)

def policy_iteration():
    #only for deterministic policy
    V = np.zeros(nrof_states, dtype=np.float32)
    policy = np.random.randint(0, 4, size=nrof_states, dtype=np.uint8)
    iteration = 1
    while True:
        print(f"\riteration {iteration}", end='')
        while True:
            delta = 0
            for state in env.unwrapped.P:
                v = V[state]  # vector
                new_v = 0
                for p, new_state, r, term in env.unwrapped.P[state][policy[state]]:
                    new_v += p * (r + gamma * V[new_state])
                V[state] = new_v
                delta = max(delta, np.abs(v - V[state]))
            if delta < EPS:
                break
        #print(V)
        #policy improvement
        a = policy.copy()
        for state in env.unwrapped.P:
            max_v = 0
            for action in env.unwrapped.P[state]:
                curV = 0
                for p, new_state, r, term in env.unwrapped.P[state][action]:
                    curV += p * (r + gamma * V[new_state])
                if curV > max_v:
                    max_v, policy[state] = curV, action
        iteration += 1
        if (a == policy).all():
            break
    policy = np.round(policy,2)
    V = np.round(V, 2)

    if PRINT:
        print("\nPolicy iteration/policy_field:")
        for i in range(0, nrof_states, 4):
            print(np.round(policy[i:i+4], 2))

        print("Policy iteration/policy_matrix:") ## зачем я это делаю?!
        _policy = np.zeros((nrof_states, nrof_action))
        _policy[np.arange(nrof_states), policy] = 1
        print(_policy)
        for row in _policy:
            print('['+', '.join(map(str,row)) + '],')


        print("Policy iteration/values_field:")
        for i in range(0, nrof_states, 4):
            print(V[i:i + 4])

        print("Policy iteration/Values_vector:")
        print(V)
    return V



def value_iteration():
    #only for deterministic policy
    V = np.zeros(nrof_states, dtype=np.float32)

    #policy evaluation
    iteration = 1
    while True:

        print(f"\riteration {iteration}", end='')
        delta = 0
        for state in env.unwrapped.P:
            v = V[state]  # vector
            max_v = 0
            for action in env.unwrapped.P[state]:
                new_v = 0
                for p, new_state, r, term in env.unwrapped.P[state][action]:
                    new_v += p * (r + gamma * V[new_state])
                if new_v > max_v:
                    max_v = new_v
            V[state] = max_v
            delta = max(delta, np.abs(v - V[state]))
        iteration += 1
        if delta < EPS:
            break
    # print(V)
    # policy improvement
    policy = np.zeros(nrof_states, dtype=np.uint8)
    for state in env.unwrapped.P:
        max_v = 0
        for action in env.unwrapped.P[state]:
            new_v = 0
            for p, new_state, r, term in env.unwrapped.P[state][action]:
                new_v += p * (r + gamma * V[new_state])
            if new_v > max_v:
                max_v, policy[state] = new_v, action

    policy = np.round(policy, 2)
    if PRINT:
        V = np.round(V, 2)

        print("\nValue iteration/policy_field:")
        for i in range(0, nrof_states, 4):
            print(policy[i:i + 4])

        print("Value iteration/policy_matrix:")  ## зачем я это делаю?!
        _policy = np.zeros((nrof_states, nrof_action))
        _policy[np.arange(nrof_states), policy] = 1
        print(_policy)


        print("Value iteration/values_field:")
        for i in range(0, nrof_states, 4):
            print(V[i:i + 4])

        print("Value iteration/Values_vector:")
        print(V)
    return V







#policy = np.zeros((nrof_states, nrof_action), dtype=np.float32) + 1/nrof_action
for state in env.unwrapped.P:
    print(f"state: {state}:")
    for action in env.unwrapped.P[state]:
        print(f"    {action}: {env.unwrapped.P[state][action]}")


if __name__=="__main__":
    policy_iteration()
    value_iteration()