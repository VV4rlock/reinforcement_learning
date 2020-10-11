import numpy as np
import pickle
import csv
import os
import matplotlib.pyplot as plt
from functools import reduce

#np.random.seed(0)
ARMS_COUNT = 10
STEP_COUNT = 1000
AVERAGING_COUNT = 10
GAMES_COUNT = 5
EXPLOTARION_EXP_COUNT = 0
INITIALIZATION = np.random.uniform(0, 1, ARMS_COUNT)
EPS = [0, 0.01, 0.1, 0.5]
MEAN_RANGE = (0, 3)
DERIVATION_RANGE = (0, 1)


means, deviations, arms = None, None, None
true_arm = None


def get_normal_func(mean, deviation):
    return lambda: np.random.normal(mean, deviation)


def generate_enviroment(mean_range=(-1, 1), deviation_range=(0, 1)):
    global means, deviations, arms, true_arm
    means = np.random.uniform(mean_range[0], mean_range[1], ARMS_COUNT)
    deviations = np.random.uniform(deviation_range[0], deviation_range[1], ARMS_COUNT)
    arms = [get_normal_func(m, d) for m, d in zip(means, deviations)]
    true_arm = np.argmax(means)
    print(f"means: {means}")
    print(f"deviationa: {deviations}")
    print(f"True arm is {true_arm}: mean: {means[true_arm]}, scale: {deviations[true_arm]}")


def eps_greedy(eps, init_estimation_value, exploration_count=0):
    estimation = np.ones(ARMS_COUNT) * init_estimation_value
    pi = np.ones(ARMS_COUNT) * eps / ARMS_COUNT

    arms_indices = np.arange(ARMS_COUNT)
    additional_prob = 1 - eps
    total_score = 0
    cum_scores = [0]
    cum_accuracy = [0]

    total_right_arms = 0
    if exploration_count:
        for i, arm in enumerate(arms):
            new_estimation = sum([arm() for _ in range(exploration_count)]) / (exploration_count - 1)
            estimation[i] = new_estimation

    choices_count = np.ones(ARMS_COUNT) # not zero because 1/n_a in new estimation

    for t in range(STEP_COUNT):
        #print(estimation)
        argmax_choice = np.argmax(estimation)

        #change probability of max estimation
        pi[argmax_choice] += additional_prob
        choice = np.random.choice(arms_indices, p=pi)
        pi[argmax_choice] -= additional_prob

        # get arm reward
        arm = arms[choice]
        av_score = sum([arm() for _ in range(AVERAGING_COUNT)]) / AVERAGING_COUNT
        total_score += av_score
        total_right_arms += 1 if choice == true_arm else 0

        # update estimation
        estimation[choice] = estimation[choice] + (av_score - estimation[choice]) / choices_count[choice]
        choices_count[choice] += 1

        cum_scores.append(total_score / (t+1))
        cum_accuracy.append(total_right_arms / (t + 1))

    return cum_scores, cum_accuracy




if __name__ == "__main__":
    generate_enviroment(MEAN_RANGE, DERIVATION_RANGE)
    ideal_mean_score = [means[true_arm] for i in range(STEP_COUNT)]

    #plt.plot(ideal_mean_score)
    plots1 = []
    plots2 = []
    for eps in EPS:
        cum_scores, cum_accuracy = eps_greedy(eps, INITIALIZATION, exploration_count=EXPLOTARION_EXP_COUNT)
        plots1.append(cum_accuracy)
        plots2.append(cum_scores)


    fig, axs = plt.subplots(2)
    for i in range(len(plots1)):
        axs[0].plot(plots2[i], label=f"eps={EPS[i]}")
        axs[1].plot(plots1[i],  label=f"eps={EPS[i]}")
    axs[0].plot(ideal_mean_score, 'm--')
    plt.show()
