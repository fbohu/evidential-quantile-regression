# Base class for all models
import tensorflow as tf
import numpy as np
import random
import time
from read_data import *
from functools import partial
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import pickle

class Model(object):
    def __init__(self, input_shape, num_neurons, num_layers, activation, patience = 50, learning_rate=3e-4, seed=0):
        self.learning_rate = learning_rate
        #self.quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        self.quantiles = [0.05, 0.95]
        self.num_quantiles = len(self.quantiles)
        self.patience = patience
        self.epochs = 1000
        self.seed = seed
        self.activation = activation

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

    def evaluate(self, x, y, y_train_mu, y_train_scale):
        preds = self.predict(x)
        y = (y*y_train_scale) + y_train_mu
        preds = (preds*y_train_scale) + y_train_mu

        tl = self.loss(y, preds)
    
        mu, sigma = self.get_mu_sigma(x)
        mu = (mu*y_train_scale) + y_train_mu
        sigma = sigma*y_train_scale

        nll_ = 0.0
        for i, q in enumerate(self.quantiles):
            nll_ += self.nll(y, tf.expand_dims(mu[:,i],1), tf.expand_dims(sigma[:,i],1), q)
        
        preds = self.predict(x)
        tic = time.time()
        # 20 times to get the average time
        for _ in range(20):
            preds = self.predict(x)
        time_ = (time.time()-tic)/20.0

        
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


    def fit_with(self, dropout, lr, batch_size):
        (x_train, y_train), (x_test, y_test), _, _  = load_dataset(self.dataset, return_as_tensor=False, verbose=False)
        
        scores = []
        for _ in range(5):
            model = self.create_model(x_train.shape[1:], 128, 2, dropout, activation=self.activation)
            optimizer = tf.optimizers.Adam(lr)
            loss_fn = self.loss_ if self.name == 'evidental' else self.nll_loss
            model.compile(optimizer=optimizer, loss=loss_fn)
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience,   restore_best_weights=True, verbose=1)
            
            history = model.fit(x_train, y_train, batch_size=int(batch_size), verbose=0, epochs=self.epochs,
                                            shuffle=True, callbacks=[callback], validation_split=0.10)
            
            scores.append(history.history['val_loss'][-1])
        #print('Test loss:', scores)
        if np.isnan(np.mean(scores)):
            return -100.0
        # Return the accuracy.
        return -1.0*np.mean(scores)
        
    def bayes_opt(self, dataset, results_path):
        self.dataset = dataset    
        params_nn ={
                'dropout':(0.1, 0.5),
                'lr':(1e-5, 5e-3),
                'batch_size':(16, 128)}
                
        np.random.seed(self.seed)
        random.seed(self.seed)
        tf.random.set_seed(self.seed)
        optimizer = BayesianOptimization(
                     f=self.fit_with,
                    pbounds=params_nn,
                    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                    random_state=123,
                )

        acquisition_function = UtilityFunction(kind="ei", xi=1e-3)
        #optimizer.maximize(init_points=50, n_iter=50)
        optimizer.maximize(init_points=15, n_iter=15, acquisition_function=acquisition_function)
        for i, res in enumerate(optimizer.res):
            print("Iteration {}: \n\t{}".format(i, res))

        print(optimizer.max)
        with open(results_path + '.pickle', 'wb') as handle:
            pickle.dump(optimizer.max, handle, protocol=pickle.HIGHEST_PROTOCOL)
