import random
import numpy as np

from csv import reader
from sklearn.ensemble import RandomForestRegressor


def load_history():
    '''
    Load past data from totally randomized search
    :param:
    :return:
    '''

    with open(history_path) as csv:

        r = reader(csv)
        # r.next()

        for line in r:

            params = line[:-1]
            score = line[-1]
            history_x.append(params)
            history_y.append(score)


def optimize_nn(l_params, variance):
    '''
    Sample from a constraint gaussian
    :param l_params: current best
    :param variance: variance of each dimension, covariance assumed zero
    :return:
    '''
    candidate = np.random.randn(6)

    candidate = np.round((candidate * variance) + l_params)

    # set the candidates to boredrlines
    if candidate[0] < 10: candidate[0] = 10
    elif candidate[0] > 30: candidate[0] = 30

    if candidate[1] < 20: candidate[1] = 20
    elif candidate[1] > 50: candidate[1] = 50

    if candidate[2] < 1: candidate[2] = 1
    elif candidate[2] > 5: candidate[2] = 5

    if candidate[3] < 4: candidate[3] = 4
    elif candidate[3] > 10: candidate[3] = 10

    if candidate[4] < 1: candidate[4] = 1
    elif candidate[4] > 5: candidate[4] = 6

    if candidate[5] < 4: candidate[5] = 4
    elif candidate[5] > 10: candidate[5] = 10

    return candidate


def gen_point(l_best_params, variance, pool_size, epsilon):
    '''
    Generate next hyper-param candidate
    :param clf: model
    :param l_best_params: current best
    :param clf: current model
    :pool_size: how many sample evaluate before returning best candidate
    :param epsilon: exploration vs exploitation
    :return:
    '''
    if random.random()<epsilon:
        return optimize_nn(l_best_params, variance)
    else:
        next = optimize_nn(l_best_params, variance)
        score_next = clf.predict(next)
        for i in range(pool_size):
            cand = optimize_nn(l_best_params, variance)
            score_cand = clf.predict(cand)
            if score_cand>score_next:
                next = cand
                score_next = score_cand
        return next


history_path = "./data/history/history_binary_glove_300x43.csv"
history_x = []
history_y = []
load_history()
clf = RandomForestRegressor(n_estimators=30)
clf.fit(history_x, history_y)
