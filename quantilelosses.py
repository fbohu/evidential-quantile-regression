import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

def NIG_NLL(y, gamma, v, alpha, beta, w_i_dis, quantile, reduce=True):
    tau_two = 2.0/(quantile*(1.0-quantile))
    twoBlambda = 2.0*2.0*beta*(1.0+tau_two*w_i_dis.mean()*v)

    nll = 0.5*tf.math.log(np.pi/v)  \
        - alpha*tf.math.log(twoBlambda)  \
        + (alpha+0.5) * tf.math.log(v*(y-gamma)**2 + twoBlambda)  \
        + tf.math.lgamma(alpha)  \
        - tf.math.lgamma(alpha+0.5)

    return tf.reduce_mean(nll) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5*(a1-1)/b1 * (v2*tf.square(mu2-mu1))  \
        + 0.5*v2/v1  \
        - 0.5*tf.math.log(tf.abs(v2)/tf.abs(v1))  \
        - 0.5 + a2*tf.math.log(b1/b2)  \
        - (tf.math.lgamma(a1) - tf.math.lgamma(a2))  \
        + (a1 - a2)*tf.math.digamma(a1)  \
        - (b1 - b2)*a1/b1
    return KL


def tilted_loss(q, e):
    return tf.maximum(q*e, (q-1)*e)


def NIG_Reg(y, gamma, v, alpha, beta, w_i_dis, quantile, omega=0.01, reduce=True, kl=False):
    tau_two = 2.0/(quantile*(1.0-quantile))
    theta = (1.0-2.0*quantile)/(quantile*(1.0-quantile))
    error = tilted_loss(quantile, y-gamma)

    w = abs(quantile-0.5)

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+alpha+1/beta
        reg = error*evi
        
    return tf.reduce_mean(reg) if reduce else reg

def quant_evi_loss(y_true, gamma, v, alpha, beta, quantile, coeff=1.0, reduce=True):
    tfd = tfp.distributions
    theta = (1.0-2.0*quantile)/(quantile*(1.0-quantile))
    mean_ = beta/(alpha-1)

    w_i_dis = tfd.Exponential(rate=1/mean_)
    mu = gamma + theta*w_i_dis.mean()
    loss_nll = NIG_NLL(y_true, mu, v, alpha, beta, w_i_dis, quantile, reduce=reduce)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta, w_i_dis, quantile, reduce=reduce)
    return loss_nll + coeff * loss_reg