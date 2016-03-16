# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:28:12 2015

@author: serizel
"""
import theano.tensor as T
from theano.ifelse import ifelse
import theano


def beta_div(X, W, H, beta):
    """Compute betat divergence"""
    div = ifelse(T.eq(beta, 2),
                 T.sum(1. / 2 * T.power(X - T.dot(H, W), 2)), 
                 ifelse(T.eq(beta, 0),
                        T.sum(X / T.dot(H, W) - T.log(X / T.dot(H, W)) - 1),
                        ifelse(T.eq(beta, 1),
                               T.sum(T.mul(X, (T.log(X) - T.log(T.dot(H, W)))) + T.dot(H, W) - X),
                                T.sum(1. / (beta * (beta - 1.)) * (T.power(X, beta) +
                                                                   (beta - 1.) *
                                                                   T.power(T.dot(H, W), beta) -
                                                                   beta *
                                                                   T.power(T.mul(X, T.dot(H, W)),
                                                                           (beta - 1)))))))
    return div


def group_div(X, W, H, beta, params):
    ind = params[0][0]
    k_cls = params[1][0]
    k_ses = params[1][1]
    lambdas = params[2]
    Sc = params[3]
    Cs = params[4]
    res_ses, up = theano.scan(fn=lambda Cs, prior_result: prior_result +\
                                            eucl_dist(W[ind, :, k_cls:k_cls+k_ses],
                                                      W[Cs, :, k_cls:k_cls+k_ses]),
                              outputs_info=T.zeros_like(beta),
                              sequences=Cs)
    sum_ses = ifelse(T.gt(Cs[0], 0), res_ses[-1], T.zeros_like(beta))
    res_cls, up = theano.scan(fn=lambda Sc, prior_result: prior_result +\
                                            eucl_dist(W[ind, :, 0:k_cls],
                                                      W[Sc, :, 0:k_cls]),
                              outputs_info=T.zeros_like(beta),
                              sequences=Sc)
    sum_cls = ifelse(T.gt(Sc[0], 0), res_cls[-1], T.zeros_like(beta))
    betaDiv = beta_div(X, W[ind].T, H, beta)

    return lambdas[0] * sum_cls + lambdas[1] * sum_ses + betaDiv, betaDiv, sum_cls, sum_ses

def noise_div(X, W, Wn, H, beta, params):
    ind = params[0][0]
    k_cls = params[1][0]
    k_ses = params[1][1]
    lambdas = params[2]
    Sc = params[3]
    sum_ses = eucl_dist(W[ind, :, k_cls:k_cls+k_ses],
                        Wn[:, k_cls:k_cls+k_ses])
    res_cls, up = theano.scan(fn=lambda Sc, prior_result: prior_result +\
                                            eucl_dist(W[ind, :, 0:k_cls],
                                                      W[Sc, :, 0:k_cls]),
                              outputs_info=T.zeros_like(beta),
                              sequences=Sc)
    sum_cls = ifelse(T.gt(Sc[0], 0), res_cls[-1], T.zeros_like(beta))
    betaDiv = beta_div(X, W[ind].T, H, beta)

    return lambdas[0] * sum_cls + lambdas[1] * sum_ses + betaDiv, betaDiv, sum_cls, sum_ses
   
def cls_sum(W, params):
    k_cls = params[0][0]
    Sc = params[1]
    res_cls, up = theano.scan(fn=lambda Sc, prior_result: prior_result +\
                                            W[Sc, :, 0:k_cls],
                              outputs_info=T.zeros_like(W[0, :, 0:k_cls]),
                              sequences=Sc)                       

    return res_cls[-1] 
    
def ses_sum (W, params):
    k_cls = params[0][0]
    k_ses = params[0][1]
    Cs = params[1]
    res_ses, up = theano.scan(fn=lambda Cs, prior_result: prior_result +\
                                            W[Cs, :, k_cls:k_cls+k_ses],
                              outputs_info=T.zeros_like(W[0, :, k_cls:k_ses+k_cls]),
                              sequences=Cs)


    return res_ses[-1]

def eucl_dist(X, Y):
    """euclidean distance"""
    return T.sum((1. /2) * (T.power(X, 2) + T.power(Y, 2) - 2 * T.mul(X, Y)))
