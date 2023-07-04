import h5py
import argparse
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from models.evidentalconv import ConvEvidental
from models.ensembleconv import ConvEnsemble
from models.dropoutconv import ConvDropout
from quantilelosses import *

def standardize(data):
    mu = data.mean(axis=0, keepdims=1)
    scale = data.std(axis=0, keepdims=1)
    scale[scale<1e-10] = 1.0

    data = (data - mu) / scale
    return data, mu, scale

def get_model(which):
    return {
        'dropout': ConvDropout,
        'ensemble': ConvEnsemble,
        'evidental': ConvEvidental
    }[which]


def main(args):
    train = h5py.File("data/depth/depth_train.h5", "r")
    test = h5py.File("data/depth/depth_test.h5", "r")
    ood = h5py.File('data/depth/apolloscape_test.h5', 'r')
    x_train, y_train = (train["image"], train["depth"])
    x_test, y_test = (test["image"], test["depth"])
    ood_x, ood_y = (ood["image"], ood["depth"])

    seeds = args.seed
    np.random.seed(seeds)
    random.seed(seeds)
    tf.random.set_seed(seeds)

    modeltype = get_model(args.model)
    
    model = modeltype(input_shape=x_train.shape[1:], 
                num_neurons= 128, 
                num_layers=3, 
                lam=0.0,
                activation='leaky_relu',
                drop_prob=0.1,
                learning_rate=3e-4,
                patience=250)


    x_train, x_train_mu, x_train_scale = standardize(x_train[:])
    x_test = (x_test[:] - x_train_mu) / x_train_scale

    y_train, y_train_mu, y_train_scale  = standardize(y_train[:])
    y_test = (y_test[:] - y_train_mu) / y_train_scale

    ood_x = (ood_x[:] - x_train_mu) / x_train_scale
    ood_y = (ood_y[:] - y_train_mu) / y_train_scale

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)


    model.train(x_train=x_train, y_train=y_train, batch_size=128, epochs=1)

    dept_res = {
        'in_dist': model.predict(x_test).numpy(),
        'out_dist': model.predict(ood_x).numpy(),
        'in_dist_unc': model.get_uncertainties(x_test).numpy(),
        'out_dist_unc': model.get_uncertainties(ood_x).numpy()
    }

    results_path = "results/depth/"
    print(dept_res)
    with open(results_path + args.model + '.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='evidental')
    parser.add_argument('--seed', type=int, default = 1)

    args = parser.parse_args()
    print(args)
    main(args)