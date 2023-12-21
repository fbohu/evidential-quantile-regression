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
from models.faithevidental import FaithEvidental
import numpy as np
import tensorflow as tf


def plot_dis(x_plot, model,x_train_mu, x_train_scale, y_train_scale, name):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10})
    width = 239.39438
    plt.rcParams["figure.figsize"] = (6,2)
    epistemic_ = model.get_uncertainties(x_plot)
    mu, aleatoric_ = model.get_mu_sigma(x_plot)
    
    true_noise = (5.0 * tf.exp(-1.5 * np.abs(x_plot)))/y_train_scale    
    #scale epistemic to be between 0 and 1
    epistemic = epistemic_[:,0]

    aleatoric = tf.expand_dims((mu[:,1]-mu[:,0]),1)

    x_plot = x_plot*x_train_scale + x_train_mu
    plt.plot(x_plot, epistemic_[:,0]+epistemic_[:,1],'--', label='Epsitemic Uncertainty - 5th Quantile', color='#4285F9')
    #plt.plot(x_plot, epistemic_[:,1],'--', label='Epsitemic Uncertainty - 95th Quantile', color='#F4B410')
    plt.plot(x_plot, aleatoric, label='Aleatoric Uncertainty', color='#0F9D50')
    #plt.plot(x_plot, true_noise, label='true', color='black')

    # add gray area outside of training data


    plt.plot([-4, -4], [0, 100], 'k--', alpha=0.4, zorder=0)
    plt.plot([+4, +4], [0, 100], 'k--', alpha=0.4, zorder=0)

    # fill gray for x < -4
    plt.fill_between([-8, -4], [-150, -150], [150, 150], color='gray', alpha=0.2)
    plt.fill_between([4, 8], [-150, -150], [150, 150], color='gray', alpha=0.2)
    
    plt.ylim(-0.05, 1.1)
    plt.gca().set_xlim(-7, 7)
    plt.legend(loc="upper left")
    plt.savefig('figures/' + name + '_combined_1.pdf', bbox_inches='tight')
    plt.close()
        #scale epistemic to be between 0 and 1
    epistemic = (epistemic_[:,0]+epistemic_[:,1])
    epistemic = tf.where(mu[:,1] > mu[:,0], epistemic/(mu[:,1]-mu[:,0]), epistemic/0.01)
    #epistemic /= abs(mu[:,1]-mu[:,0])
    #epistemic = (epistemic - np.min(epistemic))/(np.max(epistemic) - np.min(epistemic))

    aleatoric = (tf.expand_dims(aleatoric_[:,0], 1)+tf.expand_dims(aleatoric_[:,1], 1))*y_train_scale
    #aleatoric /= tf.expand_dims((mu[:,1]-mu[:,0]),1)
    #aleatoric = (aleatoric - np.min(aleatoric))/(np.max(aleatoric) - np.min(aleatoric))

    plt.plot(x_plot, epistemic, label='epistemic')
    #plt.plot(x_plot,  epistemic_[:,0], label='epistemic')
    #plt.plot(x_plot,  epistemic_[:,1], label='epistemic')
    plt.plot(x_plot, aleatoric, label='aleatoric')

    plt.plot([-4, -4], [0, 100], 'k--', alpha=0.4, zorder=0)
    plt.plot([+4, +4], [0, 100], 'k--', alpha=0.4, zorder=0)
        # fill gray for x < -4
    plt.fill_between([-8, -4], [-150, -150], [150, 150], color='gray', alpha=0.2)
    plt.fill_between([4, 8], [-150, -150], [150, 150], color='gray', alpha=0.2)
    plt.ylim(-0.05, None)
    plt.legend(loc="upper left")
    plt.savefig('figures/' + name + '_unc_1.pdf', bbox_inches='tight')
    plt.close()

    #scale epistemic to be between 0 and 1
    epistemic = epistemic_[:,0]
    epistemic = (epistemic - np.min(epistemic))/(np.max(epistemic) - np.min(epistemic))

    aleatoric = tf.expand_dims(aleatoric_[:,0], 1)*y_train_scale
    aleatoric /= tf.expand_dims((mu[:,1]-mu[:,0]),1)
    #aleatoric = (aleatoric - np.min(aleatoric))/(np.max(aleatoric) - np.min(aleatoric))

    #plt.plot(x_plot, epistemic_[:,0], label='epistemic')
    plt.plot(x_plot, epistemic_[:,1], label='epistemic')
    #plt.plot(x_plot, tf.expand_dims(aleatoric_[:,0], 1), label='aleatoric')
    plt.plot(x_plot, tf.expand_dims(aleatoric_[:,1], 1)*y_train_scale, label='aleatoric')

    plt.plot([-4, -4], [0, 1], 'k--', alpha=0.4, zorder=0)
    plt.plot([+4, +4], [0, 1], 'k--', alpha=0.4, zorder=0)
    #plt.ylim(-0.05, 1.)
    plt.legend(loc="upper left") 
    plt.savefig('figures/' + name + '_unc_2.pdf', bbox_inches='tight')
    plt.close()

    plt.plot(x_plot, tf.expand_dims(aleatoric_[:,0], 1), label='aleatoric')
    plt.plot(x_plot, tf.expand_dims(aleatoric_[:,1], 1), label='aleatoric')

    plt.plot([-4, -4], [0, 1], 'k--', alpha=0.4, zorder=0)
    plt.plot([+4, +4], [0, 1], 'k--', alpha=0.4, zorder=0)
    plt.ylim(-0.05, 0.1)
    plt.legend(loc="upper left")
    plt.savefig('figures/' + name + '_unc_3.pdf', bbox_inches='tight')
    plt.close()



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
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10})
    width = 239.39438
    plt.rcParams["figure.figsize"] = (6,2)#set_size(width, 1.5)

    x_test = x_test[:, 0]
    x_train = x_train[:, 0]
    #var = np.minimum(var, 1e3)  # for visualization

    plt.figure(figsize=(6, 2))#, dpi=200)
    #plt.title("{:.2f}th quantile".format(quantile))
    plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0)#, label="Train")
    #plt.plot(x_test, y_test, 'r--', zorder=2, label="True")
    #plt.scatter(x_test, y_test, c='r',s=1., zorder=2, label="True")
    plt.plot(x_test, y_pred, color='#007cab', zorder=3, label="{:.2f}th quantile".format(quantile))
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
            label="Epistemic Uncertainty." if k == 0 else None)
    plt.gca().set_ylim(-150, 150)
    #plt.gca().set_ylim(-5, 5)
    plt.gca().set_xlim(-7, 7)
    plt.legend(loc="upper left")
    plt.savefig('figures/' + name + 'quantile_{}.pdf'.format(quantile), bbox_inches='tight')
    plt.close()


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
        'evidental': Evidental,
        'faithevidental': FaithEvidental
    }[which]


def main(args):
    seeds = args.seed
    np.random.seed(seeds)
    random.seed(seeds)
    tf.random.set_seed(seeds)

    x_train, y_train, y_quantiles_train = get_synth_data('Dis', -4, 4, n=5000, train=True)
    x_test, y_test, y_quantiles_test = get_synth_data('Dis', -4, 4, n=250, train=True)

    x_plot, y_plot, y_quantiles_plot = get_synth_data('Dis', -7, 7, n=100, train=True, quantiles=[0.05, 0.95])

    x_train, x_train_mu, x_train_scale = standardize(x_train)
    x_test = (x_test - x_train_mu) / x_train_scale

    x_plot = (x_plot - x_train_mu) / x_train_scale

    y_train, y_train_mu, y_train_scale = standardize(y_train)
    y_test = (y_test - y_train_mu) / y_train_scale
    y_plot = (y_plot - y_train_mu) / y_train_scale
    #y_train_scale = 1.0
    #y_train_mu = 1.0

    modeltype = get_model(args.model)
    #This gives decent results:
    model = modeltype(input_shape=x_train.shape[1:], 
            num_neurons= 128, 
            num_layers=3,
            lam=1e-2,
            activation='leaky_relu',
            drop_prob=0.15,
            patience=100,
            coeff=1e-1,
            learning_rate=3e-4,
            quantiles=[0.05, 0.95],)


    model.train(x_train, y_train, batch_size=128, epochs=500, verbose=1)
    plot_q(x_train, y_train, x_test, y_test, x_train_mu, x_train_scale, y_train_mu, y_train_scale, model, args.model)
    plot_q(x_train, y_train, x_plot, y_plot, x_train_mu, x_train_scale, y_train_mu, y_train_scale, model, args.model+"_plot")

    plot_dis(x_plot, model, x_train_mu, x_train_scale, y_train_scale, 'evi')


    evi_preds = model.predict(x_plot)*y_train_scale + y_train_mu
    preds, sigma = model.get_mu_sigma(x_plot)
    sigma = sigma*y_train_scale
    var_ = model.get_uncertainties(x_plot)*y_train_scale

    plt.figure(figsize=(6,2), dpi=200)
    plt.scatter(x_test*x_train_scale+x_train_mu, y_test * y_train_scale + y_train_mu, s=1.0, c='black')
    #plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0, label="Train")
    #plt.plot(x_plot*x_train_scale+x_train_mu, y_quantiles_plot[:,0], c='red', linewidth=1, linestyle='--')
    #plt.plot(x_plot*x_train_scale+x_train_mu, y_quantiles_plot[:,1], c='red', linewidth=1, linestyle='--')

    plt.plot(x_plot*x_train_scale+x_train_mu, evi_preds[:,1], c='blue', linewidth=1, label='90% Prediction Interval')
    #plt.fill_between((x_plot*x_train_scale+x_train_mu)[:,0], evi_preds[:,1]-sigma[:,1], evi_preds[:,1]+sigma[:,1], color='green', alpha=0.5)
    plt.fill_between((x_plot*x_train_scale+x_train_mu)[:,0], evi_preds[:,1]-var_[:,1], evi_preds[:,1]+var_[:,1], color='blue', alpha=0.3, label='Epistemic Uncerainty')
    #plt.fill_between((x_plot*x_train_scale+x_train_mu)[:,0], evi_preds[:,1]-2*var_[:,1], evi_preds[:,1]+2*var_[:,1], color='blue', alpha=0.1)

    plt.plot(x_plot*x_train_scale+x_train_mu, evi_preds[:,0], c='blue', linewidth=1)
    #plt.fill_between((x_plot*x_train_scale+x_train_mu)[:,0], evi_preds[:,0]-sigma[:,0], evi_preds[:,0]+sigma[:,0], color='green', alpha=0.5)
    plt.fill_between((x_plot*x_train_scale+x_train_mu)[:,0], evi_preds[:,0]-var_[:,0], evi_preds[:,0]+var_[:,0], color='blue', alpha=0.3)
    #plt.fill_between((x_plot*x_train_scale+x_train_mu)[:,0], evi_preds[:,0]-2*var_[:,0], evi_preds[:,0]+2*var_[:,0], color='blue', alpha=0.1)

    plt.plot([-4, -4], [-150, 150], 'k--', alpha=0.4, zorder=0)
    plt.plot([+4, +4], [-150, 150], 'k--', alpha=0.4, zorder=0)

    # fill gray for x < -4
    plt.fill_between([-7, -4], [-150, -150], [150, 150], color='gray', alpha=0.2)
    plt.fill_between([4, 7], [-150, -150], [150, 150], color='gray', alpha=0.2)
    plt.ylim(-150, 157)
    plt.xlim(-7, 7)
    plt.legend(loc="upper left")
    plt.savefig('qualitiv_evi_epi_synth.pdf', bbox_inches='tight')
    plt.close()

    evi_preds = model.predict(x_test)#*y_train_scale + y_train_mu
    preds, sigma = model.get_mu_sigma(x_test)
    print(preds.shape)
    print(y_test.shape)
    print((preds[:,0] < y_test[:,0]).numpy().mean())
    print((preds[:,1] < y_test[:,0]).numpy().mean())
    print(model.evaluate(x_test, y_test, y_train_mu, y_train_scale))
    print(model.evaluate(x_plot, y_plot, y_train_mu, y_train_scale))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

                    
    parser.add_argument('--noise', type=str, default='expo',
                        choices=['expo'])
    parser.add_argument('--model', type=str, default='faithevidental')
    parser.add_argument('--n_trials', type=int, default = 1)
    parser.add_argument('--seed', type=int, default = 1)

    parser.add_argument('--speed_test', type=int, default = 0)

    args = parser.parse_args()
    print(args)
    main(args)