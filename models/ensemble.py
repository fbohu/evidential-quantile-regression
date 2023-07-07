# Ensemble model which is a model class

from .model import Model
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import numpy as np


class Ensemble(Model):
    def __init__(self, input_shape, num_neurons, num_layers, activation, num_ensembles=5, drop_prob=0.1, lam=3e-4, patience = 50, learning_rate=3e-4, seed=0,
            quantiles=[0.05, 0.95]):
        super(Ensemble, self).__init__(input_shape, num_neurons, num_layers, activation, patience, learning_rate, seed, quantiles)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.name = 'ensemble'
        self.num_ensembles = num_ensembles
        self.lam = lam
        self.drop_prob = drop_prob
        # Create the ensemble of models
        self.models = [self.create_model(input_shape, num_neurons, num_layers, dropout=drop_prob, activation= activation) for _ in range(num_ensembles)] 
        # Create optimizers for each model in the ensemble
        self.optimizers = [tf.optimizers.Adam(learning_rate) for _ in range(self.num_ensembles)]
        self.history = []

    def create_model(self, input_shape, num_neurons, num_layers, dropout, activation):
        inputs = tf.keras.Input(input_shape)
        x = inputs
        for _ in range(num_layers):
            x = tf.keras.layers.Dense(num_neurons, activation=activation)(x)
            x = tf.keras.layers.Dropout(dropout)(x)
        output = tf.keras.layers.Dense(2.0*self.num_quantiles)(x)
        model = tf.keras.Model(inputs=inputs, outputs=output)

        return model

    def train(self, x_train, y_train,batch_size=128, epochs = 10):
        for (model_, optimizer_) in zip(self.models, self.optimizers):
            model_.compile(optimizer=optimizer_, loss=self.nll_loss)
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience,   restore_best_weights=True, verbose=1)
            self.history.append(model_.fit(x_train, y_train, batch_size=batch_size,verbose=0, epochs=epochs,shuffle=True, validation_split=0.10, callbacks=[callback]))

    def predict(self, x):
        predictions = []
        for i in range(self.num_ensembles):
            output = self.models[i](x)
            mu, sigma = tf.split(output, 2, axis=-1)
            predictions.append(mu)
        return tf.reduce_mean(predictions, axis=0)

    def get_uncertainties(self, x):
        predictions = []
        for model in self.models:
            predictions.append(model(x))
        return tf.math.reduce_std(predictions, axis=0)

    def get_mu_sigma(self, x):
        mu = []
        sigma = []
        for model in self.models:
            output = model(x)
            mu_, sigma_ = tf.split(output, 2, axis=-1)
            mu.append(mu_)
            sigma.append(tf.nn.softplus(sigma_) + 1e-6)
        #sigma = tf.nn.softplus(sigma) + 1e-6
        
        return tf.reduce_mean(mu, axis=0), tf.reduce_mean(sigma, axis=0)


