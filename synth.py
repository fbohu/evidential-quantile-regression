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

def plot_q(x_train, y_train, x_test,y_test, x_train_mu, x_train_scale, y_train_mu, y_train_scale, model, name):
    mu = model.predict(x_test)
    var_ = model.get_uncertainties(x_test)
    mu = (mu*y_train_scale) + y_train_mu
    var_ = var_*y_train_scale
    y_test = (y_test*y_train_scale) + y_train_mu
    y_train = (y_train*y_train_scale) + y_train_mu
    x_train = (x_train*x_train_scale) + x_train_mu
    x_test = (x_test*x_train_scale) + x_train_mu
    for i, q in enumerate(model.quantiles):
        plot_predictions(x_train, y_train, x_test, y_test, mu[:,i], var=var_[:,i], quantile=q, n_stds=4, kk=i, name=name)

def plot_predictions(x_train, y_train, x_test, y_test, y_pred, var, quantile, n_stds=4, kk=0, name='test'):
    x_test = x_test[:, 0]
    x_train = x_train[:, 0]
    #var = np.minimum(var, 1e3)  # for visualization

    plt.figure(figsize=(5, 3), dpi=200)
    plt.title("Quantile: {:.2f}".format(quantile))
    plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0, label="Train")
    #plt.plot(x_test, y_test, 'r--', zorder=2, label="True")
    plt.scatter(x_test, y_test, c='r',s=1., zorder=2, label="True")
    plt.plot(x_test, y_pred, color='#007cab', zorder=3, label="Pred")
    #plt.plot(x_test, mu_z, color='green', zorder=3, label="Pred_or")
    #plt.plot(x_test, mu_z+2*std_z, color='green', linestyle='--', zorder=3, label="Pred_or")
    plt.plot([-4, -4], [-150, 150], 'k--', alpha=0.4, zorder=0)
    plt.plot([+4, +4], [-150, 150], 'k--', alpha=0.4, zorder=0)

    for k in np.linspace(0, n_stds, 4):
        #print(var.shape)
        #print(mu.shape)
        #print((k*var).shape)
        plt.fill_between(
            x_test, (y_pred - k * var), (y_pred + k * var),
            alpha=0.3,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Unc." if k == 0 else None)
    plt.gca().set_ylim(-150, 150)
    #plt.gca().set_ylim(-5, 5)
    plt.gca().set_xlim(-7, 7)
    plt.legend(loc="upper left")
    plt.savefig('figures/' + name + 'quantile_{}.png'.format(quantile))
    plt.show()


def standardize(data):
    mu = data.mean(axis=0, keepdims=1)
    scale = data.std(axis=0, keepdims=1)
    scale[scale<1e-10] = 1.0

    data = (data - mu) / scale
    return data, mu, scale

def get_model(which):
    return {
        'dropout': Dropout,
        'ensemble': Ensemble,
        'evidental': Evidental
    }[which]


def main(args):
    x_train, y_train, y_quantiles_train = get_synth_data('Gaussian', -4, 4, n=1000, train=True)
    x_test, y_test, y_quantiles_test = get_synth_data('Gaussian', -4, 4, n=1000, train=True)

    x_plot, y_plot, y_quantiles_plot = get_synth_data('Gaussian', -7, 7, n=100, train=True)

    x_train, x_train_mu, x_train_scale = standardize(x_train)
    x_test = (x_test - x_train_mu) / x_train_scale

    x_plot = (x_plot - x_train_mu) / x_train_scale

    y_train, y_train_mu, y_train_scale = standardize(y_train)
    y_test = (y_test - y_train_mu) / y_train_scale
    y_plot = (y_plot - y_train_mu) / y_train_scale

    modeltype = get_model(args.model)

    model = modeltype(input_shape=x_train.shape[1:], 
            num_neurons= 128, 
            num_layers=3, 
            activation='leaky_relu',
            drop_prob=0.1,
            learning_rate=3e-4)

    model.train(x_train, y_train, batch_size=32, epochs=model.epochs)
    plot_q(x_train, y_train, x_test, y_test, x_train_mu, x_train_scale, y_train_mu, y_train_scale, model, args.model)
    plot_q(x_train, y_train, x_plot, y_plot, x_train_mu, x_train_scale, y_train_mu, y_train_scale, model, args.model+"_plot")
    print(model.evaluate(x_test, y_test, y_train_mu, y_train_scale))
    print(model.evaluate(x_plot, y_plot, y_train_mu, y_train_scale))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

                    
    parser.add_argument('--noise', type=str, default='expo',
                        choices=['expo'])
    parser.add_argument('--model', type=str, default='evidental')
    parser.add_argument('--n_trials', type=int, default = 1)
    parser.add_argument('--seed', type=int, default = 1)

    parser.add_argument('--speed_test', type=int, default = 0)

    args = parser.parse_args()
    print(args)
    main(args)