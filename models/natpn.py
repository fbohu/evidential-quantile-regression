from natpn import NaturalPosteriorNetwork
from natpn.datasets import WineDataModule, BostonDataModule, ConcreteDataModule, PowerPlantDataModule, YachtDataModule, EnergyEfficiencyDataModule, Kin8nmDataModule, NavalDataModule, ProteinDataModule
import torch
import pytorch_lightning as pl
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
tfd = tfp.distributions


def _get_dataset(which):
    return {
        'wine': WineDataModule(),
        'boston': BostonDataModule(),
        'concrete': ConcreteDataModule(),
        'power-plant': PowerPlantDataModule(),
        'yacht': YachtDataModule(),
        'energy-efficiency': EnergyEfficiencyDataModule(),
        'kin8nm': Kin8nmDataModule(),
        'naval': NavalDataModule(),
        'protein': ProteinDataModule(),
    }[which]

class MyNatPn(object):
    def __init__(self, dataset_name, patience = 50, learning_rate=3e-4, seed=0, quantiles=[0.05, 0.95]):
        self.learning_rate = learning_rate
        self.quantiles = quantiles
        self.num_quantiles = len(self.quantiles)
        pl.seed_everything(seed)

        self.dm = _get_dataset(dataset_name)
        self.estimator = NaturalPosteriorNetwork(
                    #latent_dim=5,
                    encoder="tabular",
                    flow_num_layers=16,
                    learning_rate=learning_rate,
                    learning_rate_decay=True,
                    trainer_params=dict(max_epochs=500),
                )

    def loss(self):
        raise NotImplementedError

    def train(self, x_train, y_train, batch_size=128, epochs = 10):
        self.estimator.fit(self.dm)

    def predict(self, x, time = False):
        preds, sigma = self.estimator.model_(x)

        if time:
            return preds
        
        mu = preds.mu.detach().unsqueeze(1).numpy()
        v = preds.lambd.detach().unsqueeze(1).numpy()
        alpha = preds.alpha.detach().unsqueeze(1).numpy()
        beta = preds.beta.detach().unsqueeze(1).numpy()

        dist = tfd.Normal(loc=mu, scale=np.sqrt(beta/(alpha-1)))
        #dist = tfd.Normal(loc=mu, scale=np.sqrt(beta/(v*(alpha-1))))
        return dist.quantile(self.quantiles)
        
        #return preds.mu
        

    def get_mu_sigma(self, x):
        preds = self.predict(x)
        return preds, tf.ones_like(preds)

    def save(self, path):
        raise NotImplementedError

    def get_quantiles(self, data):
        raise NotImplementedError

    def evaluate(self, x, y, y_train_mu, y_train_scale):
        x = torch.from_numpy(x).float()
        #y = torch.from_numpy(y).float().squeeze()
        preds = self.predict(x)#.detach().numpy()
        print(preds.shape)
        print(y.shape)
        y = (y*y_train_scale) + y_train_mu
        preds = (preds*y_train_scale) + y_train_mu
        tl = self.loss(y, preds)
    
        mu, sigma = self.get_mu_sigma(x)
        mu = (mu*y_train_scale) + y_train_mu
        sigma = sigma#*y_train_scale

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
