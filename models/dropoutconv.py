# Dropout model which is a model class
from .model import Model
import functools
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
    UpSampling2D, Cropping2D, concatenate, ZeroPadding2D, SpatialDropout2D
import functools
from layers.conv2d import Conv2DNormalGamma



class ConvDropout(Model):
    def __init__(self, input_shape, num_neurons, num_layers, activation, num_ensembles=5, drop_prob=0.1,lam=3e-4, patience = 50, learning_rate=3e-4, seed=0):
        super(ConvDropout, self).__init__(input_shape, num_neurons, num_layers, activation, patience, learning_rate, seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.name = 'dropout'

        self.num_ensembles = num_ensembles
        self.drop_prob = drop_prob
        self.lam = lam
        # Create model
        self.model = self.create_model(input_shape, num_neurons, num_layers, dropout=drop_prob, activation= activation)
        # Create optimizers for each model in the ensemble
        self.optimizer = tf.optimizers.Adam(self.learning_rate) 

    def create_model(self, input_shape, num_neurons, num_layers, dropout, activation):
        concat_axis = 3
        inputs = tf.keras.layers.Input(shape=input_shape)

        Conv2D_ = functools.partial(Conv2D, activation=activation, padding='same')

        conv1 = Conv2D_(32, (3, 3), name='conv1_1')(inputs)
        conv1 = Conv2D_(32, (3, 3))(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D_(64, (3, 3))(pool1)
        conv2 = Conv2D_(64, (3, 3))(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D_(128, (3, 3))(pool2)
        conv3 = Conv2D_(128, (3, 3))(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D_(256, (3, 3))(pool3)
        conv4 = Conv2D_(256, (3, 3))(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D_(512, (3, 3))(pool4)
        conv5 = Conv2D_(512, (3, 3))(conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        ch, cw = get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = Conv2D_(256, (3, 3))(up6)
        conv6 = Conv2D_(256, (3, 3))(conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        ch, cw = get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = Conv2D_(128, (3, 3))(up7)
        conv7 = Conv2D_(128, (3, 3))(conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        ch, cw = get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = Conv2D_(64, (3, 3))(up8)
        conv8 = Conv2D_(64, (3, 3))(conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        ch, cw = get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = Conv2D_(32, (3, 3))(up9)
        conv9 = Conv2D_(32, (3, 3))(conv9)

        ch, cw = get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        conv10 = Conv2D_(4*2*self.num_quantiles, (1, 1))(conv9)
        evidential_output = Conv2DNormalGamma(2*self.num_quantiles, (1, 1))(conv10)

        model = tf.keras.models.Model(inputs=inputs, outputs=evidential_output)
        return model


    def train(self, x_train, y_train,batch_size=128, epochs = 10):
        self.model.compile(optimizer=self.optimizer, loss=self.nll_loss)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True, verbose=1)
        self.history = self.model.fit(x_train, y_train, batch_size=batch_size,verbose=2, epochs=epochs,shuffle=True, validation_split=0.10, callbacks=[callback])

    def predict(self, x):
        predictions = []
        for _ in range(self.num_ensembles):
            output = tf.stop_gradient(self.model(x, training=True))
            mu, sigma = tf.split(output, 2, axis=-1)
            predictions.append(mu)
        return tf.reduce_mean(predictions, axis=0)


    def get_uncertainties(self, x):
        predictions = []
        for _ in range(self.num_ensembles):
            predictions.append(tf.stop_gradient(self.model(x, training=True)))
        return tf.math.reduce_std(predictions, axis=0)

    def get_mu_sigma(self, x):
        mu = []
        sigma = []
        for _ in range(self.num_ensembles):
            output = self.model(x, training=True)
            mu_, sigma_ = tf.split(output, 2, axis=-1)
            mu.append(mu_)
            #sigma.append(sigma_)
            sigma.append(tf.nn.softplus(sigma_) + 1e-6)

        return tf.reduce_mean(mu, axis=0), tf.reduce_mean(sigma, axis=0)
        