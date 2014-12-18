import os
import sys
from numpy import *
import cPickle as pickle

def sqdist(a,b):
    d  = mat('')
    aa = power(a,2).sum(0).reshape((1, a.shape[1]))
    bb = power(b,2).sum(0).reshape((1, b.shape[1]))

    ab = dot(a.T,b)
    #d = kron( ones((1, bb.shape[1])), aa.T) + kron( ones((aa.shape[0], 1)), bb) - 2*ab
    d = repmat( aa.T, 1, bb.shape[1] ) + repmat( bb, aa.shape[0], 1) - 2*ab

    return d


def OptProjectionFast(K, S, a0, cn):
    (n, m) = K.shape
    cost   = zeros( (1, cn+2), float )
    y = dot(K, a0) 
    y = 2*(1+exp(-1*y))**-1 - 1
    cost[0][0] = -dot(dot(y.T,S), y)
    cost[0][1] = cost[0][0]

    a1 = a0 
    delta    = zeros( (1, cn+2) )
    delta[0][0] = 0
    delta[0][1] = 1

    beta    = zeros( (1, cn+2) )
    beta[0][0] = 1

    for t in range(0, cn-1):
        alpha = (delta[0][t]-1)/delta[0][t+1]
        v = a1 + alpha*(a1-a0)
        y0 = dot(K, v)
        y1 = 2*(1+exp(-1*y0))**-1 -1
        gv = -dot(dot(y1.T, S), y1)
        ty = multiply( dot(S,y1), (ones((n,1))-y1**2) )[0] 
        dgv = -dot( K.T, ty)

        # seek beta[0]
        flag = 0
        for j in range(0,50):
            b  = 2**j * beta[0][t]
            z  = v-dgv/b
            y0 = dot(K,z)
            y1 = 2*(1+exp(-1*y0))**-1 - 1
            gz = -dot(dot(y1.T, S), y1)
            dif= z-v 
            gvz= gv + dot(dgv.T, dif) + dot(b*dif.T, dif)/2

            if gz <= gvz:
                flag = 1
                beta[0][t+1] = b
                a0 = a1 
                a1 = z
                cost[0][t+2] = gz
                break

        if flag == 0:
            t = t-1
            break
        else:
            delta[0][t+2] = (1+sqrt(1+4*delta[0][t+1]**2))/2

        if abs( cost[0][t+2] - cost[0][t+1])/n <= 1e-2:
            break

    a = a1 
    cost[0] = cost[0]/n
    print cost[0].shape
    return a, cost[0]

def repmat( data, r, c ):
    return kron( ones(( r, c)), data)


class KSH(object):

    def __init__(self):
        print('Init the ksh...')
        pkl = file( '../data/data.pkl', 'rb' )
        Train, Test, self.label_index, self.sample = pickle.load(pkl)
        pkl.close()
        self.traingnd  = Train[0]
        self.traindata = Train[1]
        self.testgnd   = Train[0]
        self.testdata  = Train[1]

        # number of anchors
        self.m = 100
        # number of labeled training samples
        self.trn = 200
        

    def ksh(self, bit_num):
        # kernel computing
        anchor = self.traindata[self.sample][0]
        KTrain = sqdist(self.traindata.T, anchor.T)
        sigma  = mean(KTrain, axis=1).mean(axis=0)
        KTrain = exp(-KTrain/(2*sigma))
        mvec   = mean(KTrain, axis=0)
        KTrain = KTrain - repmat(mvec, self.traingnd.shape[0], 1)

        #pairwise label matrix
        trngnd = self.traingnd[self.label_index.T]
        temp   = repmat(trngnd, 1, self.trn)
        S0     = -ones( (self.trn, self.trn), float)
        tep    = (temp == 0).nonzero()
        S0[tep] = 1
        S = bit_num*S0

        # projection optimization
        KK = KTrain[label_index.T]
        RM = dot(KK.T,KK)
        A1 = zeros( (self.m, bit_num), float)
        flag = zeros( (1, bit_num) )

        for rr in range(0,bit_num-1):
            print rr
            if rr > 1:
                S = S-dot(y, y.T)

            LM = dot( dot(KK.T, S), KK )
            U, V = eigh( LM, RM, eigvals_only=False)
            eigenvalue = diag(V).T
            #eigenvalue, order = eigenvalue.sort(axis=0)
            eigenvalue = msort(eigenvalue.T).T
            order = eigenvalue.argsort(axis=1)

            A1[:rr] = U[:order(0)]
            tep = (dot( A1[:rr].T, RM ), A1[:rr])

            A1[:rr] = sqrt(self.trn/tep)*A1[:rr]


            #get_vec, cost[0] = OptProjectionFast(KK, S, A[:rr], 500)


        
def test():
    #a = mat('1,2,3;1,2,3')
    #b = mat('2,3,4;2,3,4')
    #print sqdist(a,b)
    t = mat('2, 6, 1, 99, 4, 3, 6')
    print msort(t.T).T
    print t.argsort(axis=1) + 1

def test2():
    pkl = file( '../data/opt.pkl', 'rb' )
    K, S, a0 = pickle.load(pkl)
    pkl.close()

    print(OptProjectionFast(K, S, a0,500))
    


if __name__ == '__main__':
    
    #test2()

    ksh = KSH()
    ksh.ksh(2)
