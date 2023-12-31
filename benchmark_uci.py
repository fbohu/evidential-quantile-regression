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
from models.evidental_gauss import EvidentalGauss
from models.natpn import MyNatPn
import numpy as np
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1) 


def get_hparams(dataset, model):
    if model == 'evidental_gauss':
        model = 'dropout'
    with open('hparams/' + dataset + '/' + model + '.pickle', 'rb') as handle:
        hparams = pickle.load(handle)
    return hparams['params']

def get_model(which):
    return {
        'dropout': Dropout,
        'ensemble': Ensemble,
        'evidental': Evidental,
        'evidental_gauss': EvidentalGauss,
        'natpn': MyNatPn
    }[which]


def main(args):
    tls = []
    nlls = []
    times = []
    results_path = 'results/' + args.dataset + "/"
    seeds = args.seed
    np.random.seed(seeds)
    random.seed(seeds)
    tf.random.set_seed(seeds)
    #for i in range(args.n_trials):
    (x_train, y_train), (x_test, y_test), y_train_mu, y_train_scale  = load_dataset(args.dataset, return_as_tensor=False)
    modeltype = get_model(args.model)
    
    modeltype = get_model(args.model)
    if args.model == 'natpn':
        # quick fix for natpn
        hpara = {}
        hpara['batch_size'] = 32
        model = modeltype(dataset_name=args.dataset, seed = seeds, patience = 50, learning_rate=3e-4, quantiles=[0.05, 0.95])
    else:
        hpara = get_hparams(args.dataset, args.model)
        print(hpara)
        model = modeltype(input_shape=x_train.shape[1:], 
            num_neurons=128,#int(hpara['hidden_size']), 
            num_layers=2,#int(hpara['layers']), 
            activation='leaky_relu',
            drop_prob=hpara['dropout'],
            learning_rate=hpara['lr'],
            seed=seeds)

    
    model.train(x_train, y_train, batch_size=int(hpara['batch_size']), epochs=500)

    tl, nll, time = model.evaluate(x_test, y_test, y_train_mu, y_train_scale)
    tls.append(tl)
    nlls.append(nll)
    times.append(time)

    results = {
        'tls': tls,
        'nlls': nlls,
        'times': times,
    }
    print(results)
    with open(results_path + args.model + str(seeds) + '.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

                    
    parser.add_argument('--dataset', type=str, default='protein',
                        choices=['boston', 'concrete', 'energy-efficiency',
                            'kin8nm', 'naval', 'power-plant', 'protein',
                            'wine', 'yacht'])
    parser.add_argument('--model', type=str, default='natpn')
    parser.add_argument('--n_trials', type=int, default = 1)
    parser.add_argument('--seed', type=int, default = 1)

    parser.add_argument('--speed_test', type=int, default = 0)

    args = parser.parse_args()
    print(args)
    main(args)