# Base class for all models
import tensorflow as tf
import numpy as np
import time

class Model(object):
    def __init__(self, input_shape, num_neurons, num_layers, activation, patience = 50, learning_rate=3e-4):
        self.learning_rate = learning_rate
        #self.quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        self.quantiles = [0.05, 0.95]
        self.num_quantiles = len(self.quantiles)
        self.patience = patience

    def loss(self):
        raise NotImplementedError

    def create_model(self):
        raise NotImplementedError

    def train(self, data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def get_quantiles(self, data):
        raise NotImplementedError

    def evaluate(self, x, y):
        preds = self.predict(x)
        tl = self.loss(y, preds)
    
        mu, sigma = self.get_mu_sigma(x)
        nll_ = 0.0
        for i, q in enumerate(self.quantiles):
            nll_ += self.nll(y, tf.expand_dims(mu[:,i],1), tf.expand_dims(sigma[:,i],1), q)
        
        tic = time.time()
        preds = self.predict(x)
        time_ = time.time()-tic
        
        return tl.numpy(), nll_.numpy(), time_

    def nll_loss(self, y, output):
        mu, sigma = tf.split(output, 2, axis=-1)
        sigma = tf.nn.softplus(sigma) + 1e-6
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            loss += self.nll(y, tf.expand_dims(mu[:,i],1), tf.expand_dims(sigma[:,i],1), q)
        return loss


    def nll(self, y, preds, sigma, quantile):
        # compute the negative log likelihood of ALD
        first_part = tf.math.log((quantile*(1-quantile))/sigma)
        second_part = self.tilted_loss(quantile, (y-preds))/sigma
        return tf.reduce_mean(-first_part+second_part)


    def tilted_loss(self, q, e):
        return tf.maximum(q*e, (q-1)*e)

    def loss(self, y, output):
        loss = 0
        for i, q in enumerate(self.quantiles):
            e = y - tf.expand_dims(output[:,i], 1)
            loss += self.tilted_loss(q, e)
        return tf.reduce_mean(loss)

    def get_batch(self, x, y, batch_size):
        idx = np.random.choice(x.shape[0], batch_size, replace=False)
        if isinstance(x, tf.Tensor):
            x_ = x[idx,...]
            y_ = y[idx,...]
        elif isinstance(x, np.ndarray) or isinstance(x, h5py.Dataset):
            idx = np.sort(idx)
            x_ = x[idx,...]
            y_ = y[idx,...]

            x_divisor = 255. if x_.dtype == np.uint8 else 1.0
            y_divisor = 255. if y_.dtype == np.uint8 else 1.0

            x_ = tf.convert_to_tensor(x_/x_divisor, tf.float32)
            y_ = tf.convert_to_tensor(y_/y_divisor, tf.float32)
        else:
            print("unknown dataset type {} {}".format(type(x), type(y)))
        return x_, y_


    