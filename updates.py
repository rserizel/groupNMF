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
    up = ifelse(T.eq(beta, 2), (T.dot(X, W)) / (T.dot(T.dot(H, W.T), W)),
                               (T.dot(T.mul(T.power(T.dot(H, W.T), (beta - 2)), X), W)) /
                               (T.dot(T.power(T.dot(H, W.T), (beta-1)), W)))
    return T.mul(H, up)

def beta_H_Sparse(X, W, H, beta, l_sp):
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
    up = ifelse(T.eq(beta, 2), (T.dot(X, W)) / (T.dot(T.dot(H, W.T), W) +
                                                l_sp),
                               (T.dot(T.mul(T.power(T.dot(H, W.T),
                                            (beta - 2)), X), W)) /
                               (T.dot(T.power(T.dot(H, W.T), (beta-1)), W) +
                                l_sp))
    return T.mul(H, up)

def beta_H_groupSparse(X, W, H, beta, l_sp, start, stop):
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
    results, _ = theano.scan(fn=lambda start_i, stop_i, prior_results, H:
                             T.set_subtensor(
                                prior_results[:, start_i:stop_i].T,
                                H[:, start_i:stop_i].T /
                                H[:, start_i:stop_i].norm(2, axis=1)).T,
                             outputs_info=T.zeros_like(H),
                             sequences=[start, stop],
                             non_sequences=H)
    cst = results[-1]
    up = ifelse(T.eq(beta, 2), (T.dot(X, W)) / (T.dot(T.dot(H, W.T), W) +
                                                l_sp * cst),
                               (T.dot(T.mul(T.power(T.dot(H, W.T),
                                            (beta - 2)), X), W)) /
                               (T.dot(T.power(T.dot(H, W.T), (beta-1)), W) +
                                l_sp * cst))
    return T.mul(H, up)


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
    up = ifelse(T.eq(beta, 2), (T.dot(X.T, H)) / (T.dot(T.dot(H, W.T).T, H)),
                               (T.dot(T.mul(T.power(T.dot(H, W.T), (beta - 2)), X).T, H)) /
                               (T.dot(T.power(T.dot(H, W.T), (beta-1)).T, H)))
    return T.mul(W, up)


def H_beta_sub(X, W, Wsub, H, Hsub, beta):
    """Update group activation with beta divergence

    Parameters
    ----------
    X : Theano tensor
        data
    W : Theano tensor
        Bases
    Wsub : Theano tensor
        group Bases        
    H : Theano tensor
        activation matrix
    Hsub : Theano tensor
        group activation matrix
    beta : Theano scalar

    Returns
    -------
    H : Theano tensor
        Updated version of the activations
    """
    up = ifelse(T.eq(beta, 2), (T.dot(X, Wsub)) / (T.dot(T.dot(H, W.T), Wsub)),
                (T.dot(T.mul(T.power(T.dot(H, W.T), (beta - 2)), X), Wsub)) /
                (T.dot(T.power(T.dot(H, W.T), (beta-1)), Wsub)))
    return T.mul(Hsub, up)
    
def W_beta_sub(X, W, Wsub, H, Hsub, beta):
    """Update group activation with beta divergence

    Parameters
    ----------
    X : Theano tensor
        data
    W : Theano tensor
        Bases
    Wsub : Theano tensor
        group Bases        
    H : Theano tensor
        activation matrix
    Hsub : Theano tensor
        group activation matrix
    beta : Theano scalar

    Returns
    -------
    H : Theano tensor
        Updated version of the activations
    """
    up = ifelse(T.eq(beta, 2), (T.dot(X.T, Hsub)) / (T.dot(T.dot(H, W.T).T, Hsub)),
                (T.dot(T.mul(T.power(T.dot(H, W.T), (beta - 2)), X).T, Hsub)) /
                (T.dot(T.power(T.dot(H, W.T), (beta-1)).T, Hsub)))
    return T.mul(Wsub, up)
    
def W_beta_sub_withcst(X, W, Wsub, H, Hsub, beta, sum_grp, lambda_grp, card_grp):
    """Update group activation with beta divergence

    Parameters
    ----------
    X : Theano tensor
        data
    W : Theano tensor
        Bases
    Wsub : Theano tensor
        group Bases        
    H : Theano tensor
        activation matrix
    Hsub : Theano tensor
        group activation matrix
    beta : Theano scalar

    Returns
    -------
    H : Theano tensor
        Updated version of the activations
    """
    up = ifelse(T.eq(beta, 2), (T.dot(X.T, Hsub) + lambda_grp * sum_grp) /
                               (T.dot(T.dot(H, W.T).T, Hsub) + lambda_grp * card_grp * Wsub),
                (T.dot(T.mul(T.power(T.dot(H, W.T), (beta - 2)), X).T, Hsub)+
                 lambda_grp * sum_grp) /
                (T.dot(T.power(T.dot(H, W.T), (beta-1)).T, Hsub) +
                 lambda_grp * card_grp * Wsub))
    return T.mul(Wsub, up)

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
    up_cls = H_beta_sub(X, W, W[:, 0:k_cls], H[start:stop, :], H[start:stop, 0:k_cls], beta)
    up_ses = H_beta_sub(X, W, W[:, k_cls:k_ses+k_cls], H[start:stop, :], H[start:stop, k_cls:k_ses+k_cls], beta)
    up_res = H_beta_sub(X, W, W[:, k_ses+k_cls:], H[start:stop, :], H[start:stop, k_ses+k_cls:], beta)
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
    up_cls = ifelse(T.gt(cardSc, 0), up_cls_with_cst, up_cls_without_cst)
    up_ses = ifelse(T.gt(cardCs, 0), up_ses_with_cst, up_ses_without_cst)
    up_res = W[ind, :, k_ses+k_cls:]*((T.dot(T.mul(T.power(T.dot(H, W[ind].T),
                                                           (beta - 2)),
                                                   X).T,
                                             H[:, k_ses+k_cls:])) /
                                      (T.dot(T.power(T.dot(H, W[ind].T),
                                                     (beta-1)).T,
                                             H[:, k_ses+k_cls:])))
    return T.concatenate((up_cls, up_ses, up_res), axis=1)

def noise_W(X, W, Wn, H, beta, params):
    """Group udpate for the bases with beta divergence

    Parameters
    ----------
    X : Theano tensor
        data
    W : Theano tensor
        Bases
    Wn : Theano tensor
         Noise bases
    H : Theano tensor
        activation matrix
    beta : Theano scalar
    params : array
        params[0][0] : indice of the group to update (corresponding to a unique couple (spk,ses))
        params[5][0] : cardSc number of elements in Sc
        params[1][0] : k_cls number of vectors in the spk bases
        params[1][1] : k_ses number of vectors in the session bases
        params[2] : [lambda1, lambda2] wieght applied on the constraints
        params[3] : Sc, ensemble of session in which speaker c is present

    Returns
    -------
    W : Theano tensor
        Updated version of the bases
    """
    ind = params[0][0]
    cardSc = params[5][0]
    k_cls = params[1][0]
    k_ses = params[1][1]
    lambdas = params[2]
    Sc = params[3]

    res_cls, up_cls = theano.scan(fn=lambda Sc, prior_result: prior_result + W[Sc, :, 0:k_cls],
                                  outputs_info=T.zeros_like(W[0, :, 0:k_cls]),
                                  sequences=Sc)
    sum_cls = res_cls[-1]
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
    up_ses = W[ind, :, k_cls:k_ses+k_cls]*((T.dot(T.mul(T.power(T.dot(H, W[ind].T),
                                                                         (beta - 2)),
                                                                 X).T,
                                                           H[:, k_cls:k_ses+k_cls]) +
                                                     lambdas[1] * Wn[:, k_cls:k_ses+k_cls]) /
                                                    (T.dot(T.power(T.dot(H, W[ind].T),
                                                                   (beta-1)).T,
                                                           H[:, k_cls:k_ses+k_cls]) +
                                                     lambdas[1] *
                                                     W[ind, :, k_cls:k_ses+k_cls]))
    up_cls = ifelse(T.gt(cardSc, 0), up_cls_with_cst, up_cls_without_cst)
    up_res = W[ind, :, k_ses+k_cls:]*((T.dot(T.mul(T.power(T.dot(H, W[ind].T),
                                                           (beta - 2)),
                                                   X).T,
                                             H[:, k_ses+k_cls:])) /
                                      (T.dot(T.power(T.dot(H, W[ind].T),
                                                     (beta-1)).T,
                                             H[:, k_ses+k_cls:])))
    return T.concatenate((up_cls, up_ses, up_res), axis=1)

def group_W_nosum(X, W, H, sum_cls, sum_ses, beta, params):
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
        params[3] : Class distance
        params[4] : session distance

    Returns
    -------
    W : Theano tensor
        Updated version of the bases
    """
    ind = params[0][0]
    cardSc = params[3][0]
    cardCs = params[3][1]
    k_cls = params[1][0]
    k_ses = params[1][1]
    lambdas = params[2]


    up_cls = W[ind, :, 0:k_cls]*((T.dot(T.mul(T.power(T.dot(H, W[ind].T),
                                                               (beta - 2)),
                                                       X).T,
                                                 H[:, 0:k_cls]) +
                                           lambdas[0] * sum_cls) /
                                          (T.dot(T.power(T.dot(H, W[ind].T),
                                                         (beta-1)).T,
                                                 H[:, 0:k_cls]) +
                                           lambdas[0] * cardSc * W[ind, :, 0:k_cls]))

    up_ses = W[ind, :, k_cls:k_ses+k_cls]*((T.dot(T.mul(T.power(T.dot(H, W[ind].T),
                                                                         (beta - 2)),
                                                                 X).T,
                                                           H[:, k_cls:k_ses+k_cls]) +
                                                     lambdas[1] * sum_ses) /
                                                    (T.dot(T.power(T.dot(H, W[ind].T),
                                                                   (beta-1)).T,
                                                           H[:, k_cls:k_ses+k_cls]) +
                                                     lambdas[1] * cardCs *
                                                     W[ind, :, k_cls:k_ses+k_cls]))

    up_res = W[ind, :, k_ses+k_cls:]*((T.dot(T.mul(T.power(T.dot(H, W[ind].T),
                                                           (beta - 2)),
                                                   X).T,
                                             H[:, k_ses+k_cls:])) /
                                      (T.dot(T.power(T.dot(H, W[ind].T),
                                                     (beta-1)).T,
                                             H[:, k_ses+k_cls:])))
                                       
    return T.concatenate((up_cls, up_ses, up_res), axis=1)
