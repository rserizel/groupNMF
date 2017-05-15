# -*- coding: utf-8 -*-
"""
beta\_nmf.py
~~~~~~~~~~~

.. topic:: Contents

  The beta_nmf module includes the beta\_nmf class,
  fit function and theano functions to compute updates and cost.
  
  Copyright 2014-2016 Romain Serizel

  This software is distributed under the terms of the GNU Public License 
  version 3 (http://www.gnu.org/licenses/gpl.txt)"""

import time
import numpy as np
import theano
import base
import updates
import costs


class BetaNMF(object):
    """BetaNMF class

    Performs nonnegative matrix factorization with Theano.
    L1-sparsity and group sparsity constraints can be applied on activations.

    Parameters
    ----------
    data_shape : tuple composed of integers
        the shape of the data to approximate

    n_components : positive integer (default 50)
        the number of latent components for the NMF model

    beta : arbitrary float (default 2)
        the beta-divergence to consider, particular cases of interest are
         * beta=2 : Euclidean distance
         * beta=1 : Kullback Leibler
         * beta=0 : Itakura-Saito

    n_iter : Positive integer (default 100)
        number of iterations

    fixed_factors : array (default Null)
        list of factors that are not updated
            e.g. fixed_factors = [0] -> H is not updated

            fixed_factors = [1] -> W is not updated

    l_sparse : Float (default 0.)
        sparsity constraint

    sparse_idx : Array
        boundaries of the groups for group sparisty [start, stop]

    verbose : Integer
        the frequence at which the score should be computed and displayed
        (number of iterations between each computation)


    Attributes
    ----------
    factors : list of arrays

        The estimated factors (factors[0] = H)"""

    # Constructor
    def __init__(self, data_shape, n_components=50, beta=2, n_iter=100,
                 fixed_factors=None, verbose=0,
                 l_sparse=0., sparse_idx=None):
        self.data_shape = data_shape
        self.n_components = n_components
        self.n_components = np.asarray(n_components, dtype='int32')
        self.beta = theano.shared(np.asarray(beta, theano.config.floatX),
                                  name="beta")
        self.verbose = verbose
        self.n_iter = n_iter
        self.scores = []
        if fixed_factors is None:
            fixed_factors = []
        self.fixed_factors = fixed_factors
        fact_ = [base.nnrandn((dim, self.n_components)) for dim in data_shape]
        self.w = theano.shared(fact_[1].astype(theano.config.floatX),
                               name="W", borrow=True, allow_downcast=True)
        self.h = theano.shared(fact_[0].astype(theano.config.floatX),
                               name="H", borrow=True, allow_downcast=True)
        self.factors = [self.h, self.w]
        self.x = theano.shared(
          np.zeros((data_shape)).astype(theano.config.floatX), name="X")
        self.eps = theano.shared(np.asarray(1e-10, theano.config.floatX),
                                 name="eps")

        self.l_sparse = theano.shared(
          np.asarray(l_sparse, theano.config.floatX),
          name="l_sparse")
        if self.l_sparse.get_value() > 0:
            if sparse_idx is None:
                self.sparse_idx = None
            else:
                self.sparse_idx = theano.shared(
                  sparse_idx.astype(theano.config.floatX),
                  name="sparse_idx")
        self.get_updates_functions()
        self.get_div_function()

    def fit(self, data):
        """Learns NMF model

        Parameters
        ----------
        X : ndarray with nonnegative entries
            The input array
        """

        self.x.set_value(data.astype(theano.config.floatX))

        print 'Fitting NMF model with %d iterations....' % self.n_iter

        # main loop
        for it in range(self.n_iter):
            if 'tick' not in locals():
                tick = time.time()
            if self.verbose > 0:
                if it == 0:
                    score = self.score()
                    print ('Iteration %d / %d, duration=%.1fms, cost=%f'
                           % (it, self.n_iter, (time.time() - tick) * 1000,
                              score))
            if 1 not in self.fixed_factors:
                self.train_w()
            if 0 not in self.fixed_factors:
                self.train_h()
            if self.verbose > 0:
                if (it+1) % self.verbose == 0:
                    score = self.score()
                    print ('Iteration %d / %d, duration=%.1fms, cost=%f'
                           % (it+1, self.n_iter, (time.time() - tick) * 1000,
                              score))
                    tick = time.time()
        print 'Done.'

    def get_div_function(self):
        """Compile the theano-based divergence function"""
        self.div = theano.function(inputs=[],
                                   outputs=costs.beta_div(self.x,
                                                          self.w.T,
                                                          self.h,
                                                          self.beta),
                                   name="div",
                                   allow_input_downcast=True)

    def get_updates_functions(self):
        """Compile the theano based update functions"""
        print "Standard rules for beta-divergence"
        if self.l_sparse.get_value() == 0:
            h_update = updates.beta_H(
              self.x,
              self.w,
              self.h,
              self.beta,
              self.eps)
        else:
            if self.sparse_idx is None:
                h_update = updates.beta_H_Sparse(self.x,
                                                 self.w,
                                                 self.h,
                                                 self.beta,
                                                 self.l_sparse,
                                                 self.eps)
            else:
                h_update = updates.beta_H_groupSparse(self.x,
                                                      self.w,
                                                      self.h,
                                                      self.beta,
                                                      self.l_sparse,
                                                      self.sparse_idx[0, ],
                                                      self.sparse_idx[1, ],
                                                      self.eps)
        w_update = updates.beta_W(self.x,
                                  self.w,
                                  self.h,
                                  self.beta,
                                  self.eps)
        self.train_h = theano.function(inputs=[],
                                       outputs=[],
                                       updates={self.h: h_update},
                                       name="trainH",
                                       allow_input_downcast=True)
        self.train_w = theano.function(inputs=[],
                                       outputs=[],
                                       updates={self.w: w_update},
                                       name="trainW",
                                       allow_input_downcast=True)

    def score(self):
        """Compute factorisation score

        Returns
        -------
        out : Float
            factorisation score"""
        return self.div()
