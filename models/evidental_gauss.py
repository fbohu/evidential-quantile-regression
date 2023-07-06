# Evidental Regression model class based on Gaussian loss
# extends the evidental class

from .model import Model
from layers.dense import *
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import numpy as np
import time
from gaussianlosses import *

class EvidentalGauss(Model):
    def __init__(self, input_shape, num_neurons, num_layers, activation, drop_prob=0.1, lam = 3e-4, patience = 50, learning_rate=3e-4, seed=0):
        super(EvidentalGauss, self).__init__(input_shape, num_neurons, num_layers, activation, patience, learning_rate, seed)
        self.name = 'evidental_gauss'
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
        output = DenseNormalGamma(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=output)

        return model

    def train(self, x_train, y_train, batch_size=128, epochs = 10):
        self.model.compile(optimizer=self.optimizer, loss=self.EvidentialRegressionLoss)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=False, verbose=1)
        self.history = self.model.fit(x_train, y_train, batch_size=batch_size, verbose=1, epochs=epochs,
                                        shuffle=True, validation_split=0.10, callbacks=[callback])

        # Custom loss function to handle the custom regularizer coefficient
    def EvidentialRegressionLoss(self, true, pred):
        return EvidentialRegression(true, pred, coeff=1e-2)