# Evidental Regression which is a model class
from .model import Model
from layers.dense import *
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import numpy as np
import time
from quantilelosses import *
import functools
import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
    UpSampling2D, Cropping2D, concatenate, ZeroPadding2D, SpatialDropout2D
import functools
from layers.conv2d import Conv2DNormalGamma



class ConvEvidental(Model):
    def __init__(self, input_shape, num_neurons, num_layers, activation, drop_prob=0.1, lam = 3e-4, patience = 50, learning_rate=3e-4,  coeff=5e-1, seed=0,
                quantiles=[0.05, 0.95]):
        super(ConvEvidental, self).__init__(input_shape, num_neurons, num_layers, activation, patience, learning_rate, seed, quantiles)
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
        concat_axis = 3
        inputs = tf.keras.layers.Input(shape=input_shape)

        Conv2D_ = functools.partial(Conv2D, activation=activation, padding='same', kernel_regularizer=l2(self.lam))

        conv1 = Conv2D_(32, (3, 3))(inputs)
        conv1 = Conv2D_(32, (3, 3))(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        pool1 = SpatialDropout2D(self.drop_prob)(pool1)

        conv2 = Conv2D_(64, (3, 3))(pool1)
        conv2 = Conv2D_(64, (3, 3))(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        pool2 = SpatialDropout2D(self.drop_prob)(pool2)

        conv3 = Conv2D_(128, (3, 3))(pool2)
        conv3 = Conv2D_(128, (3, 3))(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        pool3 = SpatialDropout2D(self.drop_prob)(pool3)

        conv4 = Conv2D_(256, (3, 3))(pool3)
        conv4 = Conv2D_(256, (3, 3))(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        pool4 = SpatialDropout2D(self.drop_prob)(pool4)

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
        conv10 = Conv2D_(4*self.num_quantiles, (1, 1))(conv9)
        evidential_output = Conv2DNormalGamma(self.num_quantiles, (1, 1))(conv10)

        model = tf.keras.models.Model(inputs=inputs, outputs=evidential_output)
        return model

    def train(self, x_train, y_train, batch_size=128, epochs = 10):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_,  metrics=[self.eval_conv, self.tilt, self.tilt2])
        callback = tf.keras.callbacks.EarlyStopping(monitor='eval_conv', patience=self.patience, restore_best_weights=True, verbose=1)
        self.history = self.model.fit(x_train, y_train, batch_size=batch_size, verbose=2, epochs=epochs,
                                        shuffle=True, validation_split=0.1, callbacks=[callback])
        #self.model.load_weights('checkpoint/evidental.h5')

    def predict(self, x):
        output = self.model(x)
        mu, _, _, _ = tf.split(output, 4, axis=-1)
        return mu

    def loss_(self, y, output):
        loss = 0.0
        mu, v, alpha, beta = tf.split(output, 4, axis=-1)
        for i, q in enumerate(self.quantiles):
            loss += quant_evi_loss(y, tf.expand_dims(mu[:,:,:,i], 3), tf.expand_dims(v[:,:,:,i], 3),
                                        tf.expand_dims(alpha[:,:,:,i], 3), tf.expand_dims(beta[:,:,:,i], 3), q, 
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
            loss += self.nll(y, tf.expand_dims(mu[:,:,:,i], 3), tf.expand_dims(sigma[:,:,:,i], 3), q)
        return loss

    def eval_conv(self, y, y_pred):
        mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
        loss = 0
        for i, q in enumerate(self.quantiles):
            loss += self.tilted_loss(q, y-tf.expand_dims(mu[:,:,:,i], 3))

        return tf.reduce_mean(loss)

    def tilt(self, y, y_pred):
        mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
        q5 = tf.expand_dims(tf.convert_to_tensor((mu[:,:,:,0]), dtype=tf.float32), 3)
        return tf.reduce_mean(tf.cast(q5 < y, dtype=tf.float32))

    def tilt2(self, y, y_pred):
        mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
        q5 = tf.expand_dims(tf.convert_to_tensor((mu[:,:,:,1]), dtype=tf.float32), 3)
        return tf.reduce_mean(tf.cast(q5 < y, dtype=tf.float32))


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)
