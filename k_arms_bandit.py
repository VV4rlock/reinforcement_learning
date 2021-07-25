import numpy as np
import pickle
import csv
import os
import matplotlib.pyplot as plt
from functools import reduce
import time


ARMS_COUNT = 10
STEP_COUNT = 1000
GAMES_COUNT = 100
EXPLOTARION_EXP_COUNT = 0
INITIALIZATION = 0  # None #10 #if None then np.random.uniform(0, 0.0005, ARMS_COUNT)
EPS = [0, 0.05]
MEAN_RANGE = (0.9, 1.1)
DERIVATION_RANGE = (0, 0.2)
ADD_NOISE = False
NOISE_MEAN = 0
NOISE_DERIVATION = 1

means, deviations, arms, arms_indices, games_indices, noise = None, None, None, None, None, None
true_arms = None


def generate_enviroment(mean_range=(-1, 1), deviation_range=(0, 1)):
    global means, deviations, arms, true_arms, arms_indices, games_indices, noise
    means = np.random.uniform(mean_range[0], mean_range[1], (GAMES_COUNT, ARMS_COUNT))
    deviations = np.random.uniform(deviation_range[0], deviation_range[1], (GAMES_COUNT, ARMS_COUNT))
    arms = lambda: np.random.normal(means, deviations)
    noise = lambda: np.random.normal(NOISE_MEAN, NOISE_DERIVATION, (GAMES_COUNT, ARMS_COUNT))
    arms_indices = np.arange(ARMS_COUNT)
    games_indices = np.arange(GAMES_COUNT)
    true_arms = np.argmax(means, axis=1)
    #print(f"means: \n{means}")
    #print(f"deviationa: \n{deviations}")
    print(f"True arm is {true_arms}: max_mean: {means[games_indices, true_arms].mean()}")


def eps_greedy(eps, init_estimation_value, exploration_count=0):
    if init_estimation_value is None:
        estimation = np.random.uniform(0, 0.0005, (GAMES_COUNT, ARMS_COUNT))
    else:
        estimation = np.ones((GAMES_COUNT, ARMS_COUNT)) * init_estimation_value
    pi = np.ones((GAMES_COUNT, ARMS_COUNT)) * eps / ARMS_COUNT

    additional_prob = 1 - eps
    total_score = 0
    cum_scores = [0]
    cum_accuracy = [0]

    total_right_arms = 0
    choices_count = np.ones((GAMES_COUNT, ARMS_COUNT))
    if exploration_count:
        estimation = arms()
        for i in range(1, exploration_count):
            estimation = estimation + (arms() - estimation) / i
        choices_count *= exploration_count

    for t in range(STEP_COUNT):

        #print(estimation)
        print(f"\rstep {t}/{STEP_COUNT}",end='')
        argmax_choice = np.argmax(estimation, axis=1)

        #change probability of max estimation
        pi[games_indices, argmax_choice] += additional_prob
        choices = np.array([np.random.choice(arms_indices, p=p) for p in pi])
        pi[games_indices, argmax_choice] -= additional_prob

        # get arm reward
        if ADD_NOISE:
            scores = (arms() + noise())[games_indices, choices]
        else:
            scores = arms()[games_indices, choices]

        total_score += scores.sum()
        total_right_arms += (choices == true_arms).sum()

        # update estimation
        cur_estimation = estimation[games_indices, choices]
        estimation[games_indices, choices] = cur_estimation + (scores - cur_estimation) / choices_count[games_indices, choices]
        choices_count[games_indices, choices] += 1

        cum_scores.append(total_score / GAMES_COUNT / (t+1))
        cum_accuracy.append(total_right_arms / GAMES_COUNT / (t + 1))
    print()
    return cum_scores, cum_accuracy


def experiment(seed, axs):
    plots1 = []
    plots2 = []
    for eps in EPS:
        print(f"exploring eps={eps}")
        np.random.seed(seed)
        cum_scores, cum_accuracy = eps_greedy(eps, INITIALIZATION, exploration_count=EXPLOTARION_EXP_COUNT)
        plots1.append(cum_accuracy)
        plots2.append(cum_scores)

    for i in range(len(plots1)):
        axs[0].plot(plots2[i], label=f"eps={EPS[i]} init={INITIALIZATION}")
        axs[1].plot(plots1[i], label=f"eps={EPS[i]} init={INITIALIZATION}")



if __name__ == "__main__":
    generate_enviroment(MEAN_RANGE, DERIVATION_RANGE)
    ideal_mean_score = [means[games_indices, true_arms].mean() for i in range(STEP_COUNT)]
    seed = int(time.time())
    #plt.plot(ideal_mean_score)
    fig, axs = plt.subplots(2)
    experiment(seed, axs)
    INITIALIZATION = 5
    experiment(seed, axs)



    axs[0].plot(ideal_mean_score, 'm--', label="maximum")
    axs[0].legend()
    axs[1].legend()
    plt.show()
