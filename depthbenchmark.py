import h5py
import argparse
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import os
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

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    print(f"Random seed set as {seed}")


def main(args):
    train = h5py.File("data/depth/depth_train.h5", "r")
    test = h5py.File("data/depth/depth_test.h5", "r")
    ood = h5py.File('data/depth/apolloscape_test.h5', 'r')
    x_train, y_train = (train["image"], train["depth"])
    x_test, y_test = (test["image"], test["depth"])
    ood_x, ood_y = (ood["image"], ood["depth"])

    x_train = x_train[:]/np.float64(255.)
    x_test = x_test[:]/np.float64(255.)

    y_train = y_train[:]/np.float64(255.)
    y_test = y_test[:]/np.float64(255.)

    ood_x = ood_x[:]/np.float64(255.)
    ood_y = ood_y[:]/np.float64(255.)

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    seeds = 1 if args.model != 'ensemble' else args.seed
    set_seed(seeds)
    
    modeltype = get_model(args.model)
    
    model = modeltype(input_shape=x_train.shape[1:], 
                num_neurons= 128, 
                num_layers=3, 
                lam=0.0,
                activation='relu',
                drop_prob=0.1,
                learning_rate=5e-5,
                patience=50)

    model.train(x_train=x_train, y_train=y_train, batch_size=32, epochs=500)
    # Prepare the validation dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(128)

    ood_dataset = tf.data.Dataset.from_tensor_slices((ood_x, ood_y))
    ood_dataset = ood_dataset.batch(128)
    N = x_test.shape[0]
    preds = np.zeros((N, 128, 160, 2))
    unc = np.zeros((N, 128, 160, 2))
    sigma = np.zeros((N, 128, 160, 2))
    
    for i, (x_batch_train, y_batch_train) in enumerate(test_dataset):
        lower = i * 128
        upper = min(lower + 128, N)
        preds[lower:upper], sigma[lower:upper] = model.get_mu_sigma(x_batch_train)
        unc[lower:upper] = model.get_uncertainties(x_batch_train)

    N = ood_x.shape[0]
    ood_preds = np.zeros((N, 128, 160, 2))
    ood_unc = np.zeros((N, 128, 160, 2))
    ood_sigma = np.zeros((N, 128, 160, 2))
    for i, (x_batch_train, y_batch_train) in enumerate(ood_dataset):
        lower = i * 128
        upper = min(lower + 128, N)
        ood_preds[lower:upper], ood_sigma[lower:upper] = model.get_mu_sigma(x_batch_train)

        ood_unc[lower:upper] = model.get_uncertainties(x_batch_train)

    dept_res = {
        'in_dist_mu': preds,
        'out_dist_mu': ood_preds,
        'in_dist_sigma': sigma,
        'out_dist_sigma': ood_sigma,
        'in_dist_unc': unc,
        'out_dist_unc': ood_unc
    }

    results_path = "results/depth/" + args.model + "_" + str(seeds) + ".h5"

    hf = h5py.File(results_path, 'w')
    hf.create_dataset('in_dist_mu', data=preds, compression="gzip")
    hf.create_dataset('out_dist_mu', data=ood_preds, compression="gzip")
    hf.create_dataset('in_dist_sigma', data=sigma, compression="gzip")
    hf.create_dataset('out_dist_sigma', data=ood_sigma, compression="gzip")
    hf.create_dataset('in_dist_unc', data=unc, compression="gzip")
    hf.create_dataset('out_dist_unc', data=ood_unc, compression="gzip")
    hf.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='evidental')
    parser.add_argument('--seed', type=int, default = 1)

    args = parser.parse_args()
    print(args)
    main(args)