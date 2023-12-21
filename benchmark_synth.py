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
import numpy as np
import tensorflow as tf

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.set_soft_device_placement(True)


def standardize(data):
    mu = data.mean(axis=0, keepdims=1)
    scale = data.std(axis=0, keepdims=1)
    scale[scale<1e-10] = 1.0

    data = (data - mu) / scale
    return data, mu, scale

def get_hparams(dataset, model):
    with open('hparams/synth/' + dataset + '_' + model + '.pickle', 'rb') as handle:
        hparams = pickle.load(handle)
    return hparams['params']

def get_model(which):
    return {
        'dropout': Dropout,
        'ensemble': Ensemble,
        'evidental': Evidental,
        'evidental_gauss': EvidentalGauss
    }[which]


def main(args):
    tls = []
    nlls = []
    times = []
    results_path = 'results/synth/' + args.dataset + "_" + args.model
    quantiles = [0.05, 0.25, 0.75, 0.95]
    seeds = args.seed
    np.random.seed(seeds)
    random.seed(seeds)
    tf.random.set_seed(seeds)
    x_train, y_train, y_train_q95 = get_synth_data(args.dataset, x_min=-4,x_max= 4, n=5000, train=True, quantiles=quantiles)
    x_test, y_test, y_test_q95 = get_synth_data(args.dataset, -4, 4, n=1000, train=True, quantiles=quantiles)
    x_plot, y_plot,  _ = get_synth_data(args.dataset, -7, 7, n=100, train=True)
    x_train, x_train_mu, x_train_scale = standardize(x_train)
    x_test = (x_test - x_train_mu) / x_train_scale
    x_plot = (x_plot - x_train_mu) / x_train_scale

    y_train, y_train_mu, y_train_scale = standardize(y_train)
    y_test = (y_test - y_train_mu) / y_train_scale
    y_plot = (y_plot - y_train_mu) / y_train_scale
    
    modeltype = get_model(args.model)
    model = modeltype(input_shape=x_train.shape[1:], 
            num_neurons=256,
            num_layers=3, 
            lam=0.0,
            activation='leaky_relu',
            drop_prob=0.1,# if args.model == 'dropout' else 0.0,
            learning_rate=5e-3,
            patience=50,
            seed=seeds,
            quantiles=quantiles)
    
    model.train(x_train, y_train, batch_size=32, epochs=500)

    tl, nll, time = model.evaluate(x_test, y_test, y_train_mu, y_train_scale)
    tls.append(tl)
    nlls.append(nll)
    times.append(time)

    preds = model.predict(x_test)
    errors = (y_test_q95-(preds * y_train_scale + y_train_mu))

    mu = (preds*y_train_scale) + y_train_mu
    sigma = model.get_uncertainties(x_test)*y_train_scale

    results = {
        'tls': tls,
        'nlls': nlls,
        'times': times,
        'mu': mu.numpy(),
        'sigma': sigma,
        'errors': errors.numpy(),
        'test': y_test_q95,
        'test_mu': y_test*y_train_scale + y_train_mu,
    }
    print(results)
    with open(results_path + str(seeds) + '.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()                
    parser.add_argument('--dataset', type=str, default='Gaussian')
    parser.add_argument('--model', type=str, default='evidental')
    parser.add_argument('--n_trials', type=int, default = 1)
    parser.add_argument('--seed', type=int, default = 1)

    args = parser.parse_args()
    print(args)
    main(args)