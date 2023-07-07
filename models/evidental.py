# Evidental Regression which is a model class
from .model import Model
from layers.dense import *
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import numpy as np
import time
from quantilelosses import *


class Evidental(Model):
    def __init__(self, input_shape, num_neurons, num_layers, activation, drop_prob=0.1, lam = 3e-4, patience = 50, learning_rate=3e-4,  coeff=5e-1, seed=0,
                    quantiles=[0.05, 0.95]):
        super(Evidental, self).__init__(input_shape, num_neurons, num_layers, activation, patience, learning_rate, seed, quantiles)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.name = 'evidental'
        self.coeff = coeff
        self.lam = lam
        self.drop_prob = drop_prob
        self.model = self.create_model(input_shape, num_neurons, num_layers, dropout=self.drop_prob, activation= activation)
        # Create optimizers for each model in the ensemble
        self.optimizer = tf.optimizers.Adam(self.learning_rate) 

    def create_model(self, input_shape, num_neurons, num_layers, dropout, activation):
        inputs = tf.keras.Input(input_shape)
        x = inputs
        for _ in range(num_layers):
            x = tf.keras.layers.Dense(num_neurons, activation=activation)(x)
            x = tf.keras.layers.Dropout(dropout)(x)
        output = DenseNormalGamma(self.num_quantiles)(x)
        model = tf.keras.Model(inputs=inputs, outputs=output)

        return model

    def train(self, x_train, y_train, batch_size=128, epochs = 10):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_,  metrics=[self.nll_eval])
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience,   restore_best_weights=True, verbose=1)
        self.history = self.model.fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs,
                                        shuffle=True, validation_split=0.10, callbacks=[callback])

    def predict(self, x):
        output = self.model(x)
        mu, _, _, _ = tf.split(output, 4, axis=-1)
        return mu

    def loss_(self, y, output):
        loss = 0.0
        mu, v, alpha, beta = tf.split(output, 4, axis=-1)
        for i, q in enumerate(self.quantiles):
            loss += quant_evi_loss(y, tf.expand_dims(mu[:,i], 1), tf.expand_dims(v[:,i],1),
                                        tf.expand_dims(alpha[:,i],1), tf.expand_dims(beta[:,i],1), q, 
                                        coeff=self.coeff)#1e-2)#5e-2)#1e-2)
        return loss

    def get_uncertainties(self, x):
        output = self.model(x)
        mu, v, alpha, beta = tf.split(output, 4, axis=-1)
        var = np.sqrt((beta /(v*(alpha - 1))))
        return var

    def get_mu_sigma(self, x):
        output = self.model(x)
        mu, v, alpha, beta = tf.split(output, 4, axis=-1)
        sigma = beta/(alpha-1.0)
        return mu, sigma

    def nll_eval(self, y, y_pred):
        mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
        sigma = beta/(alpha-1.0)        
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            loss += self.nll(y, tf.expand_dims(mu[:,i],1), tf.expand_dims(sigma[:,i],1), q)
        return loss