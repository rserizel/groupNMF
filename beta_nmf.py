# -*- coding: utf-8 -*-
"""
Copyright © 2015 Telecom ParisTech, TSI
Auteur(s) : Romain Serizel
the beta_ntf module is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
You should have received a copy of the GNU LesserGeneral Public License
along with this program. If not, see <http://www.gnu.org/licenses/>."""

import time
import numpy as np
import theano
import base
import updates
import costs


class BetaNMF(object):
    """BetaNMF class

    Performs nonnegative matrix factorization with Theano.

    Parameters
    ----------
    data_shape : the shape of the data to approximate
        tuple composed of integers

    n_components : the number of latent components for the NMF model
        positive integer

    beta : the beta-divergence to consider
        Arbitrary float. Particular cases of interest are
         * beta=2 : Euclidean distance
         * beta=1 : Kullback Leibler
         * beta=0 : Itakura-Saito

    n_iter : number of iterations
        Positive integer

    Attributes
    ----------
    factors_: list of arrays
        The estimated factors
    """

    # Constructor
    def __init__(self, data_shape, n_components=50, beta=0, n_iter=50,
                 fixed_factors=None, buff_size=0, verbose=0):
        self.data_shape = data_shape
        self.n_components = n_components
        if buff_size > 0:
            self.buff_size = buff_size
        else:
            self.buff_size = data_shape[1]
        self.n_components = np.asarray(n_components, dtype='int32')
        self.beta = theano.shared(np.asarray(beta, theano.config.floatX),
                                  name="beta")
        self.verbose = verbose
        self.n_iter = n_iter
        self.scores = []
        self.fixed_factors = fixed_factors
        fact_ = [base.nnrandn((dim, self.n_components)) for dim in data_shape]
        self.w = theano.shared(fact_[1].astype(theano.config.floatX),
                               name="W", borrow=True, allow_downcast=True)
        self.h = theano.shared(fact_[0].astype(theano.config.floatX),
                               name="H", borrow=True, allow_downcast=True)
        self.factors_ = [self.h, self.w]
        self.x = theano.shared(np.zeros((data_shape)).astype(theano.config.floatX),
                               name="X")

        self.get_updates_functions()
        self.get_div_function()

    def fit(self, data):
        """Learns NMF model

        Parameters
        ----------
        X : ndarray with nonnegative entries
            The input array
        W : ndarray
            Optional ndarray that can be broadcasted with X and
            gives weights to apply on the cost function
        """

        self.x.set_value(data)

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
                self.train_w()
            if self.verbose > 0:
                if (it+1) % self.verbose == 0:
                    score = self.score()
                    print ('Iteration %d / %d, duration=%.1fms, cost=%f'
                           % (it+1, self.n_iter, (time.time() - tick) * 1000,
                              score))
                    tick = time.time()
        print 'Done.'

    def get_div_function(self):
        """ compile the theano-based divergence function"""
        self.div = theano.function(inputs=[],
                                   outputs=costs.beta_div(self.x,
                                                          self.w.T,
                                                          self.h,
                                                          self.beta),
                                   name="div",
                                   allow_input_downcast=True)

    def get_updates_functions(self):
        """compile the theano based update functions"""
        print "Standard rules for beta-divergence"
        h_update = updates.beta_H(self.x, self.w, self.h, self.beta)
        w_update = updates.beta_W(self.x, self.w, self.h, self.beta)
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
        """Return factorisation score"""
        return self.div()
