#!/usr/bin/python3.7

import numpy as np
np.random.seed(0)

class LengthModel(object):
    
    def n_classes(self):
        return 0

    def score(self, length, label):
        return 0.0

    def max_length(self):
        return np.inf


class PoissonModel(LengthModel):
    
    def __init__(self, model, max_length = 2000, renormalize = True):
        super(PoissonModel, self).__init__()
        if type(model) == str:
            self.mean_lengths = np.loadtxt(model)
        else:
            self.mean_lengths = model
        self.num_classes = self.mean_lengths.shape[0]
        self.max_len = max_length
        self.poisson = np.zeros((max_length, self.num_classes))

        # precompute normalizations for mean length model
        self.norms = np.zeros(self.mean_lengths.shape)
        if renormalize:
            self.norms = np.round(self.mean_lengths) * np.log(np.round(self.mean_lengths)) - np.round(self.mean_lengths)
            for c in range(len(self.mean_lengths)):
                logFak = 0
                for k in range(2, int(self.mean_lengths[c])+1):
                    logFak += np.log(k)
                self.norms[c] = self.norms[c] - logFak
        # precompute Poisson distribution
        self.poisson[0, :] = -np.inf # length zero can not happen
        logFak = 0
        for l in range(1, self.max_len):
            logFak += np.log(l)
            self.poisson[l, :] = l * np.log(self.mean_lengths) - self.mean_lengths - logFak - self.norms

    def n_classes(self):
        return self.num_classes

    def score(self, length, label):
        if length >= self.max_len:
            return -np.inf
        else:
            return self.poisson[length, label]


    def PMM(self,mean_lengths1,mean_lengths2,renormalize=True):
        def logadd(a, b):
            #vectorize this function:
            if a==-np.inf and b==-np.inf:
                return -np.inf
            if a >= b:
                result =  np.log(1 + np.exp(b - a)) + a
            else:
                result =  np.log(1 + np.exp(a - b)) + b
            return result
        self.mean_lengths=(mean_lengths1+mean_lengths2)/2
        self.poisson = np.zeros((self.max_len, self.num_classes))
        self.poisson1 = np.zeros((self.max_len, self.num_classes))
        self.poisson2 = np.zeros((self.max_len, self.num_classes))

        # precompute normalizations for mean length model
        self.norms1 = np.zeros(self.mean_lengths.shape)
        self.norms2 = np.zeros(self.mean_lengths.shape)
        if renormalize:
            self.norms1 = np.round(mean_lengths1) * np.log(np.round(mean_lengths1)) - np.round(mean_lengths1)
            for c in range(len(self.mean_lengths)):
                logFak = 0
                for k in range(2, int(mean_lengths1[c]) + 1):
                    logFak += np.log(k)
                self.norms1[c] = self.norms1[c] - logFak
            ####################################################################################################
            self.norms2 = np.round(mean_lengths2) * np.log(np.round(mean_lengths2)) - np.round(mean_lengths2)
            for c in range(len(self.mean_lengths)):
                logFak = 0
                for k in range(2, int(mean_lengths2[c]) + 1):
                    logFak += np.log(k)
                self.norms2[c] = self.norms2[c] - logFak
        # precompute Poisson distribution
        self.poisson1[0, :] = -np.inf  # length zero can not happen
        self.poisson2[0, :] = -np.inf  # length zero can not happen
        logFak = 0
        for l in range(1, self.max_len):
            logFak += np.log(l)
            self.poisson1[l, :] = l * np.log(mean_lengths1) - mean_lengths1 - logFak - self.norms1
            self.poisson2[l, :] = l * np.log(mean_lengths2) - mean_lengths2 - logFak - self.norms2
            for c in range(len(self.mean_lengths)):
                self.poisson[l,c]=logadd(self.poisson1[l, c],self.poisson2[l, c])

    def score_half_poisson(self, length, label):
        if length >= self.max_len:
            return -np.inf
        if length <=self.mean_lengths[label]:
            return 0
        else:
            return self.poisson[length, label]

    def max_lengths(self):
        return self.max_len

    def half_poisson(self, length, label,alpha=0.7):
        if length <= self.mean_lengths[label]:
            return 0.0
        else:
            if alpha**(length-self.mean_lengths[label])==0:
                return -np.inf
            return np.log(alpha**(length-self.mean_lengths[label]))