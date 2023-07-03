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

def get_hparams(which):
    # load hyperparameters from hparams
    return {
        'yacht': {'learning_rate': 5e-4, 'batch_size': 1, 'layers':2},
        'naval': {'learning_rate': 5e-4, 'batch_size': 1, 'layers':2},
        'concrete': {'learning_rate': 3e-3, 'batch_size': 1, 'layers':2},
        'energy-efficiency': {'learning_rate': 2e-3, 'batch_size': 1, 'layers':2},
        'kin8nm': {'learning_rate': 1e-3, 'batch_size': 1, 'layers':2},
        'power-plant': {'learning_rate': 3e-3, 'batch_size': 2, 'layers':2},
        'boston': {'learning_rate': 1e-4, 'batch_size': 128, 'layers':2},
        'wine': {'learning_rate': 1e-4, 'batch_size': 32, 'layers':2},
        'protein': {'learning_rate': 3e-3, 'batch_size': 64, 'layers':2},
        }[which]


def get_model(which):
    return {
        'dropout': Dropout,
        'ensemble': Ensemble,
        'evidental': Evidental
    }[which]


def main(args):
    results_path = 'hparams/' + args.dataset + "/" + args.model
    seeds = args.seed
    random.seed(seeds)
    np.random.seed(seeds)
    tf.random.set_seed(seeds)
    (x_train, y_train), (x_test, y_test), y_train_mu, y_train_scale  = load_dataset(args.dataset, return_as_tensor=False)
    modeltype = get_model(args.model)
    hpara = get_hparams(args.dataset)
    
    model = modeltype(input_shape=x_train.shape[1:], num_neurons=1, 
            num_layers=1, activation='relu', learning_rate=1, seed=seeds)
    
    model.bayes_opt(args.dataset, results_path)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
                    
    parser.add_argument('--dataset', type=str, default='boston',
                        choices=['boston', 'concrete', 'energy-efficiency',
                            'kin8nm', 'naval', 'power-plant', 'protein',
                            'wine', 'yacht'])
    parser.add_argument('--model', type=str, default='evidental')
    parser.add_argument('--n_trials', type=int, default = 1)
    parser.add_argument('--seed', type=int, default = 1)

    args = parser.parse_args()
    print(args)
    main(args)