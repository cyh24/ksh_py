import cPickle as pickle
import pprint
from numpy import *
import numpy as np
import sys
import os
import random

class pkl_Pro(object):
    def __init__(self):
        print('This is pickle process class.')

    def txt2pkl(self, inTxt, dtype):
        in_file = np.loadtxt(inTxt, float)
        return in_file

    def rand(self, num, minV, maxV):
        value = []
        for i in range(0,num):
            value.append(random.randint(minV, maxV))
            
        return mat(value)


if __name__ == '__main__':
    pkl = pkl_Pro()
    train_label = pkl.txt2pkl('../data/train_label', dtype=float)
    train_data  = pkl.txt2pkl('../data/train_data', dtype=float)
    test_label   = pkl.txt2pkl('../data/test_label', dtype=float)
    test_data  = pkl.txt2pkl('../data/test_data', dtype=float)

    label_index = pkl.rand(200, 1, train_label.shape[0])
    anchor = pkl.rand( 100, 1, label_index.shape[1])

    dataSet = np.array( [ [train_label, train_data], [test_label, test_data], label_index, anchor] )


    out_pkl = file('../data/data.pkl', 'wb')
    pickle.dump( dataSet, out_pkl)
    out_pkl.close()

    #pkl = file('../data/data.pkl', 'rb')
    #train, test = pickle.load(pkl)
    #pkl.close()


