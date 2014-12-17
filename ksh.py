import os
import sys
from numpy import *
import cPickle as pickle

def sqdist(a,b):
    d = mat('')
    aa = power(a,2).sum(0)
    bb = power(b,2).sum(0)

    print a.conj().transpose().shape
    ab = a.conj().transpose()*b
    d = kron( ones((1, bb.shape[1])), aa.T) + kron( ones((aa.shape[1], 1)), bb) - 2*ab

    return d


class KSH(object):

    def __init__(self):
        print('Init the ksh...')
        pkl = file( '../data/data.pkl', 'rb' )
        Train, Test, self.label_index, self.sample = pickle.load(pkl)
        self.traingnd  = Train[0]
        self.traindata = Train[1]
        self.testgnd   = Train[0]
        self.testdata  = Train[1]
        pkl.close()

    def ksh(self):
        # kernel computing
        anchor = self.traindata[self.sample][0]
        print anchor.shape
        KTrain = sqdist(self.traindata.T, anchor.T)
        



if __name__ == '__main__':

    ksh = KSH()

    #label = mat('4, 0, 3')
    #print Test[1][label]

    ksh.ksh()
