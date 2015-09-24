# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 11:00:43 2015

@author: serizel
"""

import theano.tensor as T
import theano
from theano.ifelse import ifelse


def beta_H(X, W, H, beta):
    """Update activation with beta divergence

    Parameters
    ----------
    X : Theano tensor
        data
    W : Theano tensor
        Bases
    H : Theano tensor
        activation matrix
    beta : Theano scalar

    Returns
    -------
    H : Theano tensor
        Updated version of the activations
    """
    return H*((T.dot(T.mul(T.power(T.dot(H, W.T), (beta - 2)), X), W)) /
              (T.dot(T.power(T.dot(H, W.T), (beta-1)), W)))


def beta_W(X, W, H, beta):
    """Update bases with beta divergence

    Parameters
    ----------
    X : Theano tensor
        data
    W : Theano tensor
        Bases
    H : Theano tensor
        activation matrix
    beta : Theano scalar

    Returns
    -------
    W : Theano tensor
        Updated version of the bases
    """
    return W*((T.dot(T.mul(T.power(T.dot(H, W.T), (beta - 2)), X).T, H)) /
              (T.dot(T.power(T.dot(H, W.T), (beta-1)).T, H)))


def group_H(X, W, H, beta, params):
    """Group udpate for the activation with beta divergence

    Parameters
    ----------
    X : Theano tensor
        data
    W : Theano tensor
        Bases
    H : Theano tensor
        activation matrix
    beta : Theano scalar
    params : array
        params[0][3] : begining of the group (couple (spk, ses)) in the activation matrix
        params[0][3] : end of the group (couple (spk, ses)) in the activation matrix
        params[1][0] : k_cls number of vectors in the spk bases
        params[1][1] : k_ses number of vectors in the session bases

    Returns
    -------
    W : Theano tensor
        Updated version of the bases
    """
    k_cls = params[1][0]
    k_ses = params[1][1]
    start = params[0][3]
    stop = params[0][4]
    up_cls = H[start:stop, 0:k_cls]*((T.dot(T.mul(T.power(T.dot(H[start:stop, :], W.T),
                                                          (beta - 2)),
                                                  X),
                                            W[:, 0:k_cls])) /
                                     (T.dot(T.power(T.dot(H[start:stop, :], W.T),
                                                    (beta-1)),
                                            W[:, 0:k_cls])))
    up_ses = H[start:stop, k_cls:k_ses+k_cls]*((T.dot(T.mul(T.power(T.dot(H[start:stop, :], W.T),
                                                                    (beta - 2)),
                                                            X),
                                                      W[:, k_cls:k_ses+k_cls])) /
                                               (T.dot(T.power(T.dot(H[start:stop, :], W.T),
                                                              (beta-1)),
                                                      W[:, k_cls:k_ses+k_cls])))
    up_res = H[start:stop, k_ses+k_cls:]*((T.dot(T.mul(T.power(T.dot(H[start:stop, :], W.T),
                                                               (beta - 2)),
                                                       X),
                                                 W[:, k_ses+k_cls:])) /
                                          (T.dot(T.power(T.dot(H[start:stop, :], W.T),
                                                         (beta-1)),
                                                 W[:, k_ses+k_cls:])))
    return T.concatenate((up_cls, up_ses, up_res), axis=1)


def group_W(X, W, H, beta, params):
    """Group udpate for the bases with beta divergence

    Parameters
    ----------
    X : Theano tensor
        data
    W : Theano tensor
        Bases
    H : Theano tensor
        activation matrix
    beta : Theano scalar
    params : array
        params[0][0] : indice of the group to update (corresponding to a unique couple (spk,ses))
        params[5][0] : cardSc number of elements in Sc
        params[5][1] : cardCs number of elements in Cs
        params[1][0] : k_cls number of vectors in the spk bases
        params[1][1] : k_ses number of vectors in the session bases
        params[2] : [lambda1, lambda2] wieght applied on the constraints
        params[3] : Sc, ensemble of session in which speaker c is present
        params[4] : Cs, ensemble of speakers active in session s

    Returns
    -------
    W : Theano tensor
        Updated version of the bases
    """
    ind = params[0][0]
    cardSc = params[5][0]
    cardCs = params[5][1]
    k_cls = params[1][0]
    k_ses = params[1][1]
    lambdas = params[2]
    Sc = params[3]
    Cs = params[4]

    res_ses, up_ses = theano.scan(fn=lambda Cs, prior_result: prior_result + W[Cs, :,
                                                                               k_cls:k_ses+k_cls],
                                  outputs_info=T.zeros_like(W[0, :, k_cls:k_ses+k_cls]),
                                  sequences=Cs)
    res_cls, up_cls = theano.scan(fn=lambda Sc, prior_result: prior_result + W[Sc, :, 0:k_cls],
                                  outputs_info=T.zeros_like(W[0, :, 0:k_cls]),
                                  sequences=Sc)
    sum_cls = res_cls[-1]
    sum_ses = res_ses[-1]
    up_cls_with_cst = W[ind, :, 0:k_cls]*((T.dot(T.mul(T.power(T.dot(H, W[ind].T),
                                                               (beta - 2)),
                                                       X).T,
                                                 H[:, 0:k_cls]) +
                                           lambdas[0] * sum_cls) /
                                          (T.dot(T.power(T.dot(H, W[ind].T),
                                                         (beta-1)).T,
                                                 H[:, 0:k_cls]) +
                                           lambdas[0] * cardSc * W[ind, :, 0:k_cls]))
    up_cls_without_cst = W[ind, :, 0:k_cls]*((T.dot(T.mul(T.power(T.dot(H, W[ind].T),
                                                                  (beta - 2)),
                                                          X).T,
                                                    H[:, 0:k_cls])) /
                                             (T.dot(T.power(T.dot(H, W[ind].T),
                                                            (beta-1)).T,
                                                    H[:, 0:k_cls])))
    up_ses_with_cst = W[ind, :, k_cls:k_ses+k_cls]*((T.dot(T.mul(T.power(T.dot(H, W[ind].T),
                                                                         (beta - 2)),
                                                                 X).T,
                                                           H[:, k_cls:k_ses+k_cls]) +
                                                     lambdas[1] * sum_ses) /
                                                    (T.dot(T.power(T.dot(H, W[ind].T),
                                                                   (beta-1)).T,
                                                           H[:, k_cls:k_ses+k_cls]) +
                                                     lambdas[1] * cardCs *
                                                     W[ind, :, k_cls:k_ses+k_cls]))
    up_ses_without_cst = W[ind, :, k_cls:k_ses+k_cls]*((T.dot(T.mul(T.power(T.dot(H, W[ind].T),
                                                                            (beta - 2)),
                                                                    X).T,
                                                              H[:, k_cls:k_ses+k_cls])) /
                                                       (T.dot(T.power(T.dot(H, W[ind].T),
                                                                      (beta-1)).T,
                                                              H[:, k_cls:k_ses+k_cls])))
    up_cls = ifelse(T.gt(Sc[0], 0), up_cls_with_cst, up_cls_without_cst)
    up_ses = ifelse(T.gt(Cs[0], 0), up_ses_with_cst, up_ses_without_cst)
    up_res = W[ind, :, k_ses+k_cls:]*((T.dot(T.mul(T.power(T.dot(H, W[ind].T),
                                                           (beta - 2)),
                                                   X).T,
                                             H[:, k_ses+k_cls:])) /
                                      (T.dot(T.power(T.dot(H, W[ind].T),
                                                     (beta-1)).T,
                                             H[:, k_ses+k_cls:])))
    return T.concatenate((up_cls, up_ses, up_res), axis=1)