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
    results_path = 'hparams/synth/' + args.dataset + "_" + args.model
    seeds = args.seed
    random.seed(seeds)
    np.random.seed(seeds)
    tf.random.set_seed(seeds)
    #(x_train, y_train), (x_test, y_test), y_train_mu, y_train_scale  = load_dataset(args.dataset, return_as_tensor=False)
    x_train, y_train, y_train_q95 = get_synth_data(args.dataset, x_min=-4,x_max= 4, n=1000, train=True, quantiles=[0.05, 0.95])
    modeltype = get_model(args.model)
    
    model = modeltype(input_shape=x_train.shape[1:], num_neurons=1, 
            num_layers=1, activation='relu', learning_rate=1, seed=seeds)
    
    model.bayes_opt(args.dataset, results_path)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
                    
    parser.add_argument('--dataset', type=str, default='Gaussian')
    parser.add_argument('--model', type=str, default='evidental')
    parser.add_argument('--n_trials', type=int, default = 1)
    parser.add_argument('--seed', type=int, default = 1)

    args = parser.parse_args()
    print(args)
    main(args)