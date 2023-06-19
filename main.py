import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import argparse
import random

from read_data import *
import matplotlib.pyplot as plt

from models.ensemble import Ensemble
from models.dropout import Dropout
from models.evidental import Evidental
import numpy as np
import tensorflow as tf



def get_model(which):
    return {
        'dropout': Dropout,
        'ensemble': Ensemble,
        'evidental': Evidental
    }[which]


def main(args):
    tls = []
    nlls = []
    times = []
    results_path = 'results/'
    np.random.seed(0)
    tf.random.set_seed(0)
    for i in range(args.n_trials):
        x_train, y_train, x_test, y_test = load_boston(seed=i)
        modeltype = get_model(args.model)
        model = modeltype(input_shape=x_train.shape[1:], num_neurons=128, num_layers=3, activation='gelu', learning_rate=3e-3)

        model.train(x_train, y_train, batch_size=128, epochs=1000)

        tl, nll, time = model.evaluate(x_test, y_test)
        tls.append(tl)
        nlls.append(nll)
        times.append(time)

    results = {
        'tls': tls,
        'nlls': nlls,
        'times': times
    }
    with open(results_path + args.model + '.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='synth')
    parser.add_argument('--model', type=str, default='dropout')
    parser.add_argument('--n_trials', type=int, default = 1)

    args = parser.parse_args()
    print(args)
    main(args)