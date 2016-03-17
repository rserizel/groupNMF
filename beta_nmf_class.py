# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:02:41 2015

@author: serizel
"""

import numpy as np
import base
import time
import itertools
import more_itertools
import theano
import theano.tensor as T
import updates
import copy
import h5py
import beta_nmf
import sys
import costs

BUFF_SIZE = 20000
K_CLS = 20
K_SES = 4
K_RES = 4
N_ITER = 10
BETA = 1.


class ClassBetaNMF(object):
    """BetaNMF class

    Performs nonnegative matrix factorization with Theano.

    Parameters
    ----------
    data : array
    cls_label : array
        the class labels for the data
    ses_label : array
        the session label for the data

    buff_size : size of the buffers, adjust depending on the GPGPU memory
        integer

    n_components : the number of latent components for the NMF model
        tuple composed of integers (k_cls, k_ses, k_res)

    beta : the beta-divergence to consider
        Arbitrary float. Particular cases of interest are
         * beta=2 : Euclidean distance
         * beta=1 : Kullback Leibler
         * beta=0 : Itakura-Saito

    NMF_updates : multiplicatives rule to update NMF 'beta' or 'groupNMF'
    (default beta-NMF)
        string

    n_iter : number of iterations
        Positive integer

    lambdas : weighting factors for the constraint (default value [0, 0, 0])
        * lambda1 : constraint on class similarity
        * lambda2 : constraint on session similarity
        * lambda3 : constraint on class discrimination

    normalize : normalize the column of W
        boolean

    dist_mode : 'segment' or 'iter'
        * 'segment' the constraint distance is computed locally
          for each new segment
        * 'iter' the constraint distances are computed once at the beginning
          of the iteration


    fixed_factors : list of factors that are not updated
        array of integers with factor indices

    verbose : the frequence at which the score should be computed and displayed
        integer

    Attributes
    ----------
    data_shape : shape of the data to approximate
        tuple of intergers
    iters: fixed iterators on class and sessions

    factors_: list of arrays (theano shared variables)
        The estimated factors

    cst_dist: list of arrays (theano shared variable)
        Contains the class distances and the session distances

    X_buff: buffer for the data (theano shared variable)

    trainW and trainH: update function for the factors W and H
    (theano functions)
    """

    # Constructor
    def __init__(self, data=np.asarray([[0, 0]]), cls_label=np.asarray([0]),
                 ses_label=np.asarray([0]), buff_size=BUFF_SIZE,
                 n_components=(K_CLS, K_SES, K_RES), beta=BETA,
                 NMF_updates='beta', n_iter=N_ITER, lambdas=[0, 0, 0],
                 normalize=False, fixed_factors=None, verbose=0,
                 dist_mode='segment',Wn=None):
        self.data_shape = data.shape
        self.buff_size = np.min((buff_size, data.shape[0]))
        self.n_components = np.asarray(n_components, dtype='int32')
        self.beta = theano.shared(np.asarray(beta, theano.config.floatX),
                                  name="beta")
        self.verbose = verbose
        self.normalize = normalize
        self.lambdas = np.asarray(lambdas, dtype=theano.config.floatX)
        self.n_iter = n_iter
        self.NMF_updates = NMF_updates
        self.iters = {}
        self.scores = []
        self.dist_mode = dist_mode
        if fixed_factors is None:
            fixed_factors = []
        self.fixed_factors = fixed_factors
        fact_ = np.asarray([base.nnrandn((self.data_shape[1],
                                          np.sum(self.n_components)))
                            for i in more_itertools.unique_everseen(itertools.izip(cls_label,
                                                                                   ses_label))])
        self.W = theano.shared(fact_.astype(theano.config.floatX), name="W",
                               borrow=True, allow_downcast=True)
        fact_ = np.asarray(base.nnrandn((self.data_shape[0],
                                         np.sum(self.n_components))))
        self.H = theano.shared(fact_.astype(theano.config.floatX), name="H",
                               borrow=True, allow_downcast=True)
        self.factors_ = [self.H, self.W]
        if Wn is not None:
            self.Wn = Wn
        self.X_buff = theano.shared(np.zeros((self.buff_size,
                                              self.data_shape[1])).astype(theano.config.floatX),
                                    name="X_buff")
        if (self.NMF_updates == 'groupNMF') & (self.dist_mode == 'iter'):
            self.cls_sums = theano.shared(np.zeros((np.max(cls_label)+1,
                                                   self.data_shape[1],
                                                   self.n_components[0])
                                                   ).astype(theano.config.floatX),
                                          name="cls_sums",
                                          borrow=True,
                                          allow_downcast=True)
            self.ses_sums = theano.shared(np.zeros((np.max(ses_label)+1,
                                                   self.data_shape[1],
                                                   self.n_components[1])
                                                   ).astype(theano.config.floatX),
                                          name="ses_sums",
                                          borrow=True,
                                          allow_downcast=True)
            self.get_sum_function()
        self.get_updates_functions()
        self.get_norm_function()
        self.get_div_function()

    def average_and_select(self, comp):
        ind = []
        if 0 in comp:
            ind = np.hstack((ind, np.arange(self.n_components[0])))
        if 1 in comp:
            ind = np.hstack((ind, np.arange(self.n_components[0],
                                            self.n_components[0]+self.n_components[1])))
        if 2 in comp:
            ind = np.hstack((ind, np.arange(self.n_components[1],
                                            self.n_components[1]+self.n_components[2])))
        W_comp = self.W.get_value()[:, :, ind.astype(int)]
        n_comp = len(ind)
        W_avg = np.zeros((W_comp.shape[1], n_comp*len(set(self.iters['cls'][:, 0]))))
        for i in range(len(set(self.iters['cls'][:, 0]))):
            if W_comp[self.iters['cls'][:, 0] == i, :, :].shape[0] > 0:
                W_avg[:, i*n_comp:(i+1)*n_comp] = np.mean(W_comp[self.iters['cls'][:, 0] == i,
                                                                 :,
                                                                 :],
                                                          axis=0)
        return W_avg

    def check_segments_length(self, data, cls_label, ses_label):
        cls = []
        cls_ind = []
        for i in more_itertools.unique_everseen(itertools.izip(cls_label, ses_label)):
            cls.append(i)
            start = np.where((cls_label == i[0]) & (ses_label == i[1]))[0][0]
            stop = np.where((cls_label == i[0]) & (ses_label == i[1]))[0][-1]+1
            cls_ind.append([start, np.min([start+self.buff_size, stop])])
            if data[(cls_label == i[0]) & (ses_label == i[1]), :].shape[0] > self.buff_size:
                ind = np.where((cls_label == i[0]) & (ses_label == i[1]))[0]
                if self.verbose > 0:
                    print "segment {0} to {1} is too long (length={2}, buffer={3})"\
                          "\n please increase buffer size or segment will be truncated"\
                          .format(ind[0], ind[-1], ind[-1]-ind[0]+1, self.buff_size)
        return np.asarray(cls), np.asarray(cls_ind)

    def compute_Cs_Sc(self, cls_train, ses_train):
        Cs = []
        for j in range(int(max(ses_train)+1)):
            Cstmp = []
            for i in range(len(self.iters['cls'])):
                if self.iters['cls'][i][1] == j:
                    Cstmp.append(self.iters['cls'][i][0])
            Cs.append(Cstmp)

        Sc = []
        for j in range(int(max(cls_train)+1)):
            Sctmp = []
            for i in range(len(self.iters['cls'])):
                if self.iters['cls'][i][0] == j:
                    Sctmp.append(self.iters['cls'][i][1])
            Sc.append(Sctmp)
        return Cs, Sc

    def compute_sum_indices(self, ind, lbl):
        Sci = copy.deepcopy(self.iters['Sc'][int(lbl[0])])
        Sci.remove(int(lbl[1]))
        Csi = copy.deepcopy(self.iters['Cs'][int(lbl[1])])
        Csi.remove(int(lbl[0]))
        # sum over all Sc!=c(card Sc - 1)
        card_tot = len(self.iters['cls']) -\
                   len(set(self.iters['cls'][:, 0])) - len(Sci)

        indices = np.asarray([ind,
                              lbl[2],
                              int(lbl[3]),
                              lbl[4],
                              lbl[5],
                              int(lbl[0]),
                              int(lbl[1])], dtype='int32')
        card = np.asarray([len(Sci),
                           len(Csi),
                           card_tot], dtype='int16')
        # Sci and Csi need to be converted in indexes for W[i,:,:]
        if len(Sci) > 0:
            tmp = np.zeros((len(Sci),))
            for i in range(len(Sci)):
                tmp[i] = np.where((self.iters['cls'][:, 0] == int(lbl[0])) &
                                  (self.iters['cls'][:, 1] == Sci[i]))[0][0]
        else:
            tmp = -np.ones((1,))
        Sci = tmp.astype('int32')
        if len(Csi) > 0:
            tmp = np.zeros((len(Csi),))
            for i in range(len(Csi)):
                tmp[i] = np.where((self.iters['cls'][:, 0] == Csi[i]) &
                                  (self.iters['cls'][:, 1] == int(lbl[1])))[0][0]
        else:
            tmp = -np.ones((1,))
        Csi = tmp.astype('int32')
        return indices, Csi, Sci, card

    def fit(self, X, cls_label, ses_label):
        global_tick = time.time()
        print "Reordering data..."
        data = base.reorder_cls_ses(X, cls_label, ses_label)
        X = data['data']
        cls_label = data['cls']
        ses_label = data['ses']
        self.update_iters(X, cls_label, ses_label)
        data = base.truncate(X, cls_label, ses_label, self.iters['cls_ind'])
        X = data['data']
        cls_label = data['cls']
        ses_label = data['ses']
        ind = data['ind']
        self.iters['cls_ind'] = ind
        self.H.set_value(self.H.get_value()[0:X.shape[0], ])
        if self.normalize:
            self.normalize_W_H()

        buff = self.generate_buffer_from_lbl(X, cls_label, ses_label,
                                             random=True, truncate=True)
        if self.buff_size > X.shape[0]:
            self.X_buff.set_value(X.astype(theano.config.floatX))
        self.scores.append(self.score_buffer(X, buff))
        print 'Fitting NMF model with %d iterations....' % self.n_iter
        for it in range(self.n_iter):
            if self.dist_mode == 'iter':
                for i in range(int(np.max(cls_label)+1)):
                    Sci = np.hstack(np.where(self.iters['cls'][:,0] == i))
                    if Sci.shape[0] > 0:
                        self.class_sum(i, self.n_components, Sci)
                for i in range(int(np.max(ses_label)+1)):  
                    Csi = np.hstack(np.where(self.iters['cls'][:,1] == i))
                    if Csi.shape[0] > 0:
                        self.ses_sum(i, self.n_components, Csi)
                    
            if self.verbose > 0:
                if (it+1) % self.verbose == 0:
                    if 'tick' not in locals():
                        tick = time.time()
                    print '\n\n NMF model, iteration {0}/{1}'.format(it+1,
                                                                     self.n_iter)
            buff = self.generate_buffer_from_lbl(X, cls_label, ses_label,
                                                 random=True, truncate=True)
            self.update_buffer(X, buff, it)
            if self.normalize:
                self.normalize_W_H()
            if self.verbose > 0:
                if (it+1) % self.verbose == 0:
                    self.scores.append(self.score_buffer(X, buff))
                    if self.NMF_updates == 'beta':
                        if self.scores[-1] > 0:
                            print 'Score: %.1f' % self.score[-1]
                    if self.NMF_updates == 'groupNMF':
                        if self.scores[-1][0][0] > 0:
                            print 'Score: %.1f' % self.scores[-1][0][0]
                            print 'Beta-divergence: %.1f' % self.scores[-1][0][1]
                            print 'Class distance : %.1f (%.1f)' % (self.scores[-1][0][2]*self.lambdas[0],
                                                                    self.scores[-1][0][2])
                            print 'Session distance : %.1f (%.1f)' % (self.scores[-1][0][3]*self.lambdas[1],
                                                                      self.scores[-1][0][3])                    
                            print 'Duration=%.1fms' % ((time.time() - tick) * 1000)
                    sys.stdout.flush()
        print 'Total duration=%.1fms' % ((time.time() - global_tick) * 1000)

    def generate_buffer_from_lbl(self,
                                 X,
                                 cls_label,
                                 ses_label,
                                 random=False,
                                 truncate=False):
        buff_fill = 0
        buff_lbl = []
        buff_ind = []
        if random:
            iter = more_itertools.random_permutation(\
            more_itertools.unique_everseen(itertools.izip(cls_label,
                                                          ses_label)))
        else:
            iter = self.iters['cls']
        if self.buff_size > X.shape[0]:
            buff_ind = []
            for i in iter:
                bloc_length = X[(cls_label == i[0]) &
                                (ses_label == i[1]), :].shape[0]
                ind = np.where((cls_label == i[0]) &
                               (ses_label == i[1]))[0]
                buff_ind.append([i[0],
                                 i[1],
                                 buff_fill,
                                 buff_fill+bloc_length,
                                 ind[0],
                                 ind[-1]+1])
        else:
            for i in iter:
                bloc_length = X[(cls_label == i[0]) &
                                (ses_label == i[1]), :].shape[0]
                ind = np.where((cls_label == i[0]) &
                               (ses_label == i[1]))[0]
                if bloc_length > self.buff_size:
                    # process the current buffer first if any
                    if buff_fill > 0:
                        buff_ind.append(buff_lbl)
                    if truncate:
                        # fill a new buffer the truncated segment
                        buff_lbl = []
                        buff_lbl.append([i[0], i[1], 0, self.buff_size,
                                         ind[0], ind[0]+self.buff_size])
                        buff_ind.append(buff_lbl)
                        # empty buffer and continue processing
                        buff_fill = 0
                        buff_lbl = []
                    else:
                        bloc_left = bloc_length
                        while bloc_left > self.buff_size:
                            buff_lbl = []
                            buff_lbl.append([i[0],
                                             i[1],
                                             0,
                                             self.buff_size, ind[-1]+1 - bloc_left,
                                             ind[-1]+1 - bloc_left+self.buff_size])
                            print bloc_left, buff_lbl
                            buff_ind.append(buff_lbl)
                            bloc_left -= self.buff_size
                        buff_lbl = []
                        buff_lbl.append([i[0],
                                         i[1],
                                         0,
                                         bloc_left,
                                         ind[-1]+1 - bloc_left,
                                         ind[-1]+1])
                        buff_fill = bloc_left
                        print bloc_left, buff_lbl

                else:
                    if buff_fill + bloc_length <= self.buff_size:
                        buff_lbl.append([i[0], i[1],
                                         buff_fill,
                                         buff_fill+bloc_length,
                                         ind[0],
                                         ind[-1]+1])
                        buff_fill = buff_fill+bloc_length
                    else:
                        buff_ind.append(buff_lbl)
                        buff_lbl = []
                        buff_lbl.append([i[0],
                                         i[1],
                                         0,
                                         bloc_length,
                                         ind[0],
                                         ind[-1]+1])
                        buff_fill = bloc_length
            if buff_fill > 0:
                buff_ind.append(buff_lbl)
        return buff_ind

    def get_div_function(self):
        tind = T.ivector('ind')
        if self.NMF_updates == 'beta':
            self.div = theano.function(inputs=[tind],
                                       outputs=costs.beta_div(self.X_buff[tind[1]:tind[2], ],
                                                              self.W[tind[0]].T,
                                                              self.H[tind[3]:tind[4], ],
                                                              self.beta),
                                       name="div",
                                       allow_input_downcast=True)
        if self.NMF_updates == 'groupNMF':
            tcomp = T.ivector('comp')
            tlambda = T.fvector('lambda')
            tSc = T.ivector('Sc')
            tCs = T.ivector('Cs')
            tparams = [tind, tcomp, tlambda, tSc, tCs]
            cost, beta_div, cls_dist, ses_dist = costs.group_div(self.X_buff[tind[1]:tind[2], ],
                                                                 self.W,
                                                                 self.H[tind[3]:tind[4], ],
                                                                 self.beta,
                                                                 tparams)

            self.div = theano.function(inputs=[tind, tcomp, tlambda, tSc, tCs],
                                       outputs=[cost,
                                                beta_div,
                                                cls_dist,
                                                ses_dist],
                                       name="div",
                                       allow_input_downcast=True,
                                       on_unused_input='ignore')

        if self.NMF_updates == 'noiseNMF':
            tcomp = T.ivector('comp')
            tlambda = T.fvector('lambda')
            tSc = T.ivector('Sc')
            tparams = [tind, tcomp, tlambda, tSc]
            cost, beta_div, cls_dist, ses_dist = costs.noise_div(self.X_buff[tind[1]:tind[2], ],
                                                                 self.W,
                                                                 self.Wn,
                                                                 self.H[tind[3]:tind[4], ],
                                                                 self.beta,
                                                                 tparams)

            self.div = theano.function(inputs=[tind, tcomp, tlambda, tSc],
                                       outputs=[cost,
                                                beta_div,
                                                cls_dist,
                                                ses_dist],
                                       name="div",
                                       allow_input_downcast=True,
                                       on_unused_input='ignore')


    def get_norm_function(self):
        tind = T.ivector('ind')
        results, updates = theano.scan(fn=base.norm_col,
                                       sequences=[self.W[tind[0]].T,
                                                  self.H[tind[1]:tind[2], ].T])
        w_norm = results[0]
        h_norm = results[1]
        norm_H = T.set_subtensor(self.H[tind[1]:tind[2], ], h_norm.T)
        norm_W = T.set_subtensor(self.W[tind[0]], w_norm.T)
        self.norm_W_H = theano.function(inputs=[tind],
                                        outputs=[],
                                        updates=[(self.W, norm_W),
                                                 (self.H, norm_H)],
                                        name="norm_w_h",
                                        allow_input_downcast=True)

    def get_sum_function(self): 
        tind = T.iscalar('ind')
        tcomp = T.ivector('comp')
        tSC_ind = T.ivector('Sc')
        tparams = [tcomp, tSC_ind]                                     
        cls_sum = T.set_subtensor(self.cls_sums[tind],
                                  costs.cls_sum(self.W, tparams))
        self.class_sum = theano.function(inputs=[tind, tcomp, tSC_ind],
                                          outputs=[],
                                          updates={self.cls_sums: cls_sum},
                                          name="class_sum",
                                          allow_input_downcast=True,
                                          on_unused_input='ignore')
                                          
        ses_sum = T.set_subtensor(self.ses_sums[tind],
                                  costs.ses_sum(self.W, tparams))

        self.ses_sum = theano.function(inputs=[tind, tcomp, tSC_ind],
                                          outputs=[ses_sum],
                                          updates={self.ses_sums: ses_sum},
                                          name="ses_sum",
                                          allow_input_downcast=True,
                                          on_unused_input='ignore')
                                         

    def get_updates_functions(self):
        tind = T.ivector('ind')

        if self.NMF_updates == 'beta':
            print "Standard rules for beta-divergence"
            H_update = T.set_subtensor(self.H[tind[3]:tind[4], ],
                                       updates.beta_H(self.X_buff[tind[1]:tind[2], ],
                                                      self.W[tind[0]],
                                                      self.H[tind[3]:tind[4], ],
                                                      self.beta))
            W_update = T.set_subtensor(self.W[tind[0]],
                                       updates.beta_W(self.X_buff[tind[1]:tind[2], ],
                                                      self.W[tind[0]],
                                                      self.H[tind[3]:tind[4], ],
                                                      self.beta))
            self.trainH = theano.function(inputs=[tind],
                                          outputs=[],
                                          updates={self.H: H_update},
                                          name="trainH",
                                          allow_input_downcast=True)
            self.trainW = theano.function(inputs=[tind],
                                          outputs=[],
                                          updates={self.W: W_update},
                                          name="trainW",
                                          allow_input_downcast=True)

        if self.NMF_updates == 'groupNMF':
            tcomp = T.ivector('comp')
            tlambda = T.fvector('lambda')
            tcard = T.bvector('card')

            print "Group NMF with class specific rules for beta-divergence"
            if self.dist_mode=='iter':
                tparams = [tind, tcomp, tlambda, tcard]
                print "Compute contraint distances once per iteration" 
                H_update = T.set_subtensor(self.H[tind[3]:tind[4], ],
                                           updates.group_H(self.X_buff[tind[1]:tind[2], ],
                                                           self.W[tind[0]],
                                                           self.H,
                                                           self.beta,
                                                           tparams))
                W_update = T.set_subtensor(self.W[tind[0]],
                                           updates.group_W_nosum(self.X_buff[tind[1]:tind[2], ],
                                                           self.W,
                                                           self.H[tind[3]:tind[4], ],
                                                           self.cls_sums[tind[5]],
                                                           self.ses_sums[tind[6]],
                                                           self.beta,
                                                           tparams))
                self.trainH = theano.function(inputs=[tind,
                                                      tcomp,
                                                      tlambda,
                                                      tcard],
                                              outputs=[],
                                              updates={self.H: H_update},
                                              name="trainH",
                                              on_unused_input='ignore',
                                              allow_input_downcast=True)
                self.trainW = theano.function(inputs=[tind,
                                                      tcomp,
                                                      tlambda,
                                                      tcard],
                                              outputs=[],
                                              updates={self.W: W_update},
                                              name="trainW",
                                              on_unused_input='ignore',
                                              allow_input_downcast=True)

            else:
                print "Compute contraint distances at each segment update"  
                tSc = T.ivector('Sc')
                tCs = T.ivector('Cs')
                tparams = [tind, tcomp, tlambda, tSc, tCs, tcard]                
                H_update = T.set_subtensor(self.H[tind[3]:tind[4], ],
                                           updates.group_H(self.X_buff[tind[1]:tind[2], ],
                                                           self.W[tind[0]],
                                                           self.H,
                                                           self.beta,
                                                           tparams))
                W_update = T.set_subtensor(self.W[tind[0]],
                                           updates.group_W(self.X_buff[tind[1]:tind[2], ],
                                                           self.W,
                                                           self.H[tind[3]:tind[4], ],
                                                           self.beta,
                                                           tparams))
                self.trainH = theano.function(inputs=[tind,
                                                      tcomp,
                                                      tlambda,
                                                      tSc,
                                                      tCs,
                                                      tcard],
                                              outputs=[],
                                              updates={self.H: H_update},
                                              name="trainH",
                                              on_unused_input='ignore',
                                              allow_input_downcast=True)
                self.trainW = theano.function(inputs=[tind,
                                                      tcomp,
                                                      tlambda,
                                                      tSc,
                                                      tCs,
                                                      tcard],
                                              outputs=[],
                                              updates={self.W: W_update},
                                              name="trainW",
                                              on_unused_input='ignore',
                                              allow_input_downcast=True)
        if self.NMF_updates == 'noiseNMF':
            tcomp = T.ivector('comp')
            tlambda = T.fvector('lambda')
            tcard = T.bvector('card')

            print "Group NMF with noise reference rules for beta-divergence" 
            tSc = T.ivector('Sc')
            tCs = T.ivector('Cs')
            tparams = [tind, tcomp, tlambda, tSc, tCs, tcard]               
            H_update = T.set_subtensor(self.H[tind[3]:tind[4], ],
                                       updates.group_H(self.X_buff[tind[1]:tind[2], ],
                                                       self.W[tind[0]],
                                                       self.H,
                                                       self.beta,
                                                       tparams))
            W_update = T.set_subtensor(self.W[tind[0]],
                                       updates.noise_W(self.X_buff[tind[1]:tind[2], ],
                                                       self.W,
                                                       self.Wn,
                                                       self.H[tind[3]:tind[4], ],
                                                       self.beta,
                                                       tparams))
            self.trainH = theano.function(inputs=[tind,
                                                  tcomp,
                                                  tlambda,
                                                  tSc,
                                                  tCs,
                                                  tcard],
                                          outputs=[],
                                          updates={self.H: H_update},
                                          name="trainH",
                                          on_unused_input='ignore',
                                          allow_input_downcast=True)
            self.trainW = theano.function(inputs=[tind,
                                                  tcomp,
                                                  tlambda,
                                                  tSc,
                                                  tcard],
                                          outputs=[],
                                          updates={self.W: W_update},
                                          name="trainW",
                                          on_unused_input='ignore',
                                          allow_input_downcast=True)

    def normalize_W_H(self):
        for i in range(len(self.iters['cls'])):
            indices = np.asarray([i, self.iters['cls_ind'][i][0],
                                  self.iters['cls_ind'][i][1]],
                                 dtype='int32')
            self.norm_W_H(indices)

    def save(self, factor_list, fname='factors.h5'):
        file = h5py.File(fname)
        if 1 in factor_list:
            file.create_dataset('W', data=self.W.get_value())
        if 0 in factor_list:
            file.create_dataset('H', data=self.H.get_value())
        file.create_dataset('scores', data=self.scores)
        file.create_dataset('n_components', data=self.n_components)
        file.create_dataset('beta', data=self.beta.get_value())
        file.create_dataset('/iters/cls', data=self.iters['cls'])
        file.create_dataset('/iters/cls_ind', data=self.iters['cls_ind'])
        file.close()

    def score(self, ind, lbl):
        indices, Csi, Sci, _ = self.compute_sum_indices(ind, lbl)
        if self.NMF_updates == 'beta':
            return self.div(indices)
        if self.NMF_updates == 'groupNMF':
            return self.div(indices, self.n_components, self.lambdas, Sci, Csi)

    def score_buffer(self, data, buff_ind):
        if self.NMF_updates == 'beta':
            score = 0
        if self.NMF_updates == 'groupNMF':
            score = np.zeros((1, 4))
        if self.buff_size > data.shape[0]:
            # "Fitting all the data in the buffer..."
            # self.X_buff.set_value(data.astype(theano.config.floatX))
            for i in range(len(buff_ind)):
                ind = np.asarray(np.where((self.iters['cls'][:, 0] == buff_ind[i][0]) &
                                          (self.iters['cls'][:, 1] == buff_ind[i][1]))[0][0],
                                 dtype='int32')
                score += self.score(ind, buff_ind[i])
        else:
            for j in range(len(buff_ind)):
                buff_lbl = np.asarray(buff_ind[j], dtype='int32')
                buff = np.zeros((buff_lbl[-1][3], data.shape[1]))
                for i in range(len(buff_lbl)):
                    buff[buff_lbl[i][2]:buff_lbl[i][3]] = data[buff_lbl[i][4]:buff_lbl[i][5]]
                self.X_buff.set_value(buff.astype(theano.config.floatX))

                for i in range(len(buff_lbl)):
                    ind = np.asarray(np.where((self.iters['cls'][:, 0] == buff_lbl[i][0]) &
                                              (self.iters['cls'][:, 1] == buff_lbl[i][1]))[0][0],
                                     dtype='int32')
                    score += self.score(ind, buff_lbl[i])
        return score

    def select(self, comp):
        ind = []
        if 0 in comp:
            ind = np.hstack((ind, np.arange(self.n_components[0])))
        if 1 in comp:
            ind = np.hstack((ind,
                             np.arange(self.n_components[0],
                                       self.n_components[0]+self.n_components[1])))
        if 2 in comp:
            ind = np.hstack((ind,
                             np.arange(self.n_components[1],
                                       self.n_components[1]+self.n_components[2])))
        W_comp = self.W.get_value()[:, :, ind.astype(int)]
        W_reshape = np.zeros((W_comp.shape[1],
                              W_comp.shape[0]*W_comp.shape[2]))
        for i in range(W_comp.shape[0]):
            W_reshape[:, i*W_comp.shape[2]:(i+1)*W_comp.shape[2]] = W_comp[i, ]
        return W_reshape

    def transform(self,
                  X,
                  comp=[0, 1],
                  n_iter=None,
                  buff_size=None,
                  fname='prediction.h5',
                  dataset='',
                  average_comp=False,
                  average_act=False,
                  seg_length=625,
                  l_sparse=0,
                  sparse_idx=None):
        if n_iter == None:
            n_iter = self.n_iter
        if buff_size == None:
            buff_size = self.buff_size
        if average_comp:
            W = self.average_and_select(comp)
        else:
            W = self.select(comp)
        buff_size = buff_size/W.shape[1]
        print buff_size
        f = h5py.File(fname)
        if average_act:
            H_out = f.create_dataset("H_{0}".format(dataset),
                                     (X.shape[0]/seg_length, W.shape[1]))
            buff_size = int(np.floor(buff_size/seg_length)*seg_length)
            out_size = buff_size/seg_length
        else:
            H_out = f.create_dataset("H_{0}".format(dataset),
                                     (X.shape[0], W.shape[1]))
        nmf_pred = beta_nmf.BetaNMF((buff_size, X.shape[1]),
                                    n_components=W.shape[1],
                                    beta=self.beta.get_value(),
                                    n_iter=n_iter,
                                    fixed_factors=[1],
                                    buff_size=buff_size,
                                    verbose=self.verbose,
                                    l_sparse=l_sparse,
                                    sparse_idx=sparse_idx)
        nmf_pred.w.set_value(W.astype(theano.config.floatX))
        i = -1
        for i in range(X.shape[0]/buff_size):
            nmf_pred.data_shape = X[i*buff_size:(i+1)*buff_size, ].shape
            print "Bloc: {0}, size {1}".format(i, nmf_pred.data_shape)
            nmf_pred.h.set_value(base.nnrandn((buff_size,
                                               nmf_pred.n_components)).astype(theano.config.floatX))
            nmf_pred.fit(X[i*buff_size:(i+1)*buff_size, ])
            if average_act:
                H_out[i*out_size:(i+1)*out_size, ] = np.mean(np.reshape(nmf_pred.h.get_value(),
                                                                        (out_size,
                                                                         seg_length,
                                                                         nmf_pred.h.get_value()\
                                                                         .shape[1])),
                                                             axis=1)
            else:
                H_out[i*buff_size:(i+1)*buff_size, ] = nmf_pred.h.get_value()

        nmf_pred.data_shape = X[(i+1)*buff_size:, ].shape
        print i+1, nmf_pred.data_shape
        nmf_pred.h.set_value(base.nnrandn((nmf_pred.data_shape[0],
                                           nmf_pred.n_components)).astype(theano.config.floatX))
        nmf_pred.fit(X[(i+1)*buff_size:, ])
        if average_act:
            H_out[(i+1)*out_size:, ] = np.mean(np.reshape(nmf_pred.h.get_value(),
                                                          (H_out.shape[0]-(i+1)*out_size,
                                                           seg_length,
                                                           nmf_pred.h.get_value().shape[1])),
                                               axis=1)
        else:
            H_out[(i+1)*buff_size:, ] = nmf_pred.h.get_value()
        f.close()

    def update(self, ind, lbl):
        indices, Csi, Sci, card = self.compute_sum_indices(ind, lbl)
        if self.NMF_updates == 'beta':
            if 1 not in self.fixed_factors:
                self.trainW(indices)
            if 0 not in self.fixed_factors:
                self.trainH(indices)
        if self.NMF_updates == 'groupNMF':
            if self.dist_mode == 'segment':
                if 1 not in self.fixed_factors:
                    self.trainW(indices,
                                self.n_components,
                                self.lambdas,
                                Sci,
                                Csi,
                                card)
                if 0 not in self.fixed_factors:
                    self.trainH(indices,
                                self.n_components,
                                self.lambdas,
                                Sci,
                                Csi,
                                card)
            else:
                if 1 not in self.fixed_factors:
                    self.trainW(indices,
                                self.n_components,
                                self.lambdas,
                                card)
                if 0 not in self.fixed_factors:
                    self.trainH(indices,
                                self.n_components,
                                self.lambdas,
                                card)                


    def update_buffer(self, data, buff_ind, it):
        if self.NMF_updates == 'beta':
            score = 0
        if self.NMF_updates == 'groupNMF':
            score = np.zeros((1, 4))                
        if self.buff_size > data.shape[0]:
            # "Fitting all the data in the buffer..."
            # self.X_buff.set_value(data.astype(theano.config.floatX))
            for i in range(len(buff_ind)):
                ind = np.asarray(np.where((self.iters['cls'][:, 0] == buff_ind[i][0]) &
                                          (self.iters['cls'][:, 1] == buff_ind[i][1]))[0][0],
                                 dtype='int32')
                self.update(ind, buff_ind[i])
                if self.verbose > 0:
                    if (it+1) % self.verbose == 0:
                        score += self.score(ind, buff_ind[i])
        else:
            for j in range(len(buff_ind)):
                buff_lbl = np.asarray(buff_ind[j], dtype='int32')
                buff = np.zeros((buff_lbl[-1][3], data.shape[1]))
                for i in range(len(buff_lbl)):
                    buff[buff_lbl[i][2]:buff_lbl[i][3]] = data[buff_lbl[i][4]:buff_lbl[i][5]]
                self.X_buff.set_value(buff.astype(theano.config.floatX))

                for i in range(len(buff_lbl)):
                    ind = np.asarray(np.where((self.iters['cls'][:, 0] == buff_lbl[i][0]) &
                                              (self.iters['cls'][:, 1] == buff_lbl[i][1]))[0][0],
                                     dtype='int32')
                    self.update(ind, buff_lbl[i])

    def update_iters(self, data, cls_label, ses_label):
        cls, cls_ind = self.check_segments_length(data, cls_label, ses_label)
        self.iters.update({'cls': cls, 'cls_ind': cls_ind})
        Cs, Sc = self.compute_Cs_Sc(cls_label, ses_label)
        self.iters.update({'Cs': Cs, 'Sc': Sc})


def load(fname="factors", updates="beta"):
    f = h5py.File(fname, 'r')

    nmf = ClassBetaNMF(n_components=f['n_components'][:],
                       beta=f['beta'],
                       NMF_updates="groupNMF",
                       verbose=1)

    nmf.iters['cls'] = f['/iters/cls'][:]
    nmf.iters['cls_ind'] = f['/iters/cls_ind'][:]
    if "H" in f:
        nmf.H.set_value(f['H'][:])
    if "W" in f:
        nmf.W.set_value(f['W'][:])
    f.close()
    nmf.get_updates_functions()
    nmf.get_norm_function()
    nmf.get_div_function()

    return nmf
