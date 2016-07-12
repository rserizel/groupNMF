# -*- coding: utf-8 -*-
"""
cost.py
~~~~~~~
.. topic:: Contents

    The cost module regroups the cost functions used for the group NMF"""
import theano.tensor as T
from theano.ifelse import ifelse
import theano


def beta_div(X, W, H, beta):
    """Compute beta divergence D(X|WH)

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
    div : Theano scalar
        beta divergence D(X|WH)"""
    div = ifelse(
      T.eq(beta, 2),
      T.sum(1. / 2 * T.power(X - T.dot(H, W), 2)),
      ifelse(
        T.eq(beta, 0),
        T.sum(X / T.dot(H, W) - T.log(X / T.dot(H, W)) - 1),
        ifelse(
          T.eq(beta, 1),
          T.sum(T.mul(X, (T.log(X) - T.log(T.dot(H, W)))) + T.dot(H, W) - X),
          T.sum(1. / (beta * (beta - 1.)) * (T.power(X, beta) +
                (beta - 1.) * T.power(T.dot(H, W), beta) -
                beta * T.power(T.mul(X, T.dot(H, W)), (beta - 1)))))))
    return div


def group_div(X, W, H, beta, params):
    """Compute beta divergence D(X|WH), intra-class distance
    and intra-session distance for a particular
    (class, session) couple [1]_.


    Parameters
    ----------
    X : Theano tensor
        data
    W : Theano tensor
        Bases
    H : Theano tensor
        activation matrix
    beta : Theano scalar
    params : Theano tensor
        Matrix of parameter related to class/session.
            :params[0][0]: index for the (class, session) couple
            :params[1][0]: number of vector basis related to class
            :params[1][1]: number of vector basis related to session
            :params[2]: weight on the class/session similarity constraints
            :params[3]: sessions in which class c appears
            :params[4]: classes present in session s



    Returns
    -------
    cost : Theano scalar
        total cost
    div : Theano scalar
        beta divergence D(X|WH)
    sum_cls : Theano scalar
        intra-class distance
    sum_ses : Theano scalar
        intra-session distance"""
    ind = params[0][0]
    k_cls = params[1][0]
    k_ses = params[1][1]
    lambdas = params[2]
    Sc = params[3]
    Cs = params[4]
    res_ses, up = theano.scan(
      fn=lambda Cs,
      prior_result: prior_result + eucl_dist(
        W[ind, :, k_cls:k_cls+k_ses], W[Cs, :, k_cls:k_cls+k_ses]),
      outputs_info=T.zeros_like(beta),
      sequences=Cs)
    sum_ses = ifelse(T.gt(Cs[0], 0), res_ses[-1], T.zeros_like(beta))
    res_cls, up = theano.scan(
      fn=lambda Sc,
      prior_result: prior_result + eucl_dist(
        W[ind, :, 0:k_cls], W[Sc, :, 0:k_cls]),
      outputs_info=T.zeros_like(beta),
      sequences=Sc)
    sum_cls = ifelse(T.gt(Sc[0], 0), res_cls[-1], T.zeros_like(beta))
    betaDiv = beta_div(X, W[ind].T, H, beta)

    cost = lambdas[0] * sum_cls + lambdas[1] * sum_ses + betaDiv
    return cost, betaDiv, sum_cls, sum_ses


def noise_div(X, W, Wn, H, beta, params):
    """Compute beta divergence D(X|WH)
    noise-related distance (distance to a noise reference)
    and intra-speaker distance
    for a particular class (only one noise session considered here).

    Parameters
    ----------
    X : Theano tensor
        data
    W : Theano tensor
        Bases
    H : Theano tensor
        activation matrix
    beta : Theano scalar
    params : Theano tensor
        Matrix of parameter related to class/session.
            :params[0][0]: index for the (class, session) couple
            :params[1][0]: number of vector basis related to class
            :params[1][1]: number of vector basis related to session
            :params[2]: weight on the class/session similarity constraints
            :params[3]: sessions in which class c appears



    Returns
    -------
    cost : Theano scalar
        total cost
    div : Theano scalar
        beta divergence D(X|WH)
    sum_cls : Theano scalar
        intra-class distance
    sum_ses : Theano scalar
        distance to noise reference"""
    ind = params[0][0]
    k_cls = params[1][0]
    k_ses = params[1][1]
    lambdas = params[2]
    Sc = params[3]
    sum_ses = eucl_dist(W[ind, :, k_cls:k_cls+k_ses],
                        Wn)
    res_cls, up = theano.scan(
      fn=lambda Sc,
      prior_result: prior_result + eucl_dist(
        W[ind, :, 0:k_cls], W[Sc, :, 0:k_cls]),
      outputs_info=T.zeros_like(beta),
      sequences=Sc)
    sum_cls = ifelse(T.gt(Sc[0], 0), res_cls[-1], T.zeros_like(beta))
    betaDiv = beta_div(X, W[ind].T, H, beta)

    cost = lambdas[0] * sum_cls + lambdas[1] * sum_ses + betaDiv
    return cost, betaDiv, sum_cls, sum_ses


def cls_sum(W, params):
    """Compute summ of basis for a particular class.
    To be used in the constrained multiplicative update rules [1]_.

    Parameters
    ----------
    W : Theano tensor
        Bases
    params : Theano tensor
        Matrix of parameter related to class/session.
            :params[0][0]: number of vector basis related to class
            :params[1]: sessions in which class c appears



    Returns
    -------
    sum_cls : Theano scalar
        sum of the basis for the class"""
    k_cls = params[0][0]
    Sc = params[1]
    res_cls, up = theano.scan(
      fn=lambda Sc,
      prior_result: prior_result + W[Sc, :, 0:k_cls],
      outputs_info=T.zeros_like(W[0, :, 0:k_cls]),
      sequences=Sc)

    return res_cls[-1]


def ses_sum(W, params):
    """Compute sum of basis for a particular session.
    To be used in the constrained multiplicative update rules [1]_.

    Parameters
    ----------
    W : Theano tensor
        Bases
    params : Theano tensor
        Matrix of parameter related to class/session.
            :params[0][0]: number of vector basis related to class
            :params[1]: class that appear in session s

    Returns
    -------
    sum_ses : Theano scalar
        sum of the basis for the session"""
    k_cls = params[0][0]
    k_ses = params[0][1]
    Cs = params[1]
    res_ses, up = theano.scan(
      fn=lambda Cs,
      prior_result: prior_result + W[Cs, :, k_cls:k_cls+k_ses],
      outputs_info=T.zeros_like(W[0, :, k_cls:k_ses+k_cls]),
      sequences=Cs)
    return res_ses[-1]


def eucl_dist(X, Y):
    """Compute Euclidean distance between X and Y

    Parameters
    ----------
    X : Theano tensor
    Y : Theano tensor

    Returns
    -------
    out : Theano scalar
        Euclidean distance"""
    return T.sum((1. / 2) * (T.power(X, 2) + T.power(Y, 2) - 2 * T.mul(X, Y)))
