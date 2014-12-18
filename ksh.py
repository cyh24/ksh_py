import os
import sys
from numpy import *
import cPickle as pickle
from scipy.linalg import eigh

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
    cost   = zeros( cn+2, float )
    y = dot(K, a0) 
    y = 2*(1+exp(-1*y))**-1 - 1
    cost[0] = -dot(dot(y.T,S), y)
    cost[1] = cost[0]

    a1 = a0 
    delta    = zeros(cn+2)
    delta[0] = 0
    delta[1] = 1

    beta    = zeros(cn+2)
    beta[0] = 1

    for t in range(0, cn-1):
        alpha = (delta[t]-1)/delta[t+1]
        v = a1 + alpha*(a1-a0)
        y0 = dot(K, v)
        y1 = 2*(1+exp(-1*y0))**-1 -1
        gv = -dot(dot(y1.T, S), y1)
        ty = multiply( dot(S,y1), (ones((n,1))-y1**2) )[0] 
        dgv = -dot( K.T, ty)

        # seek beta[0]
        flag = 0
        for j in range(0,50):
            b  = 2**j * beta[t]
            z  = v-dgv/b
            y0 = dot(K,z)
            y1 = 2*(1+exp(-1*y0))**-1 - 1
            gz = -dot(dot(y1.T, S), y1)
            dif= z-v 
            gvz= gv + dot(dgv.T, dif) + dot(b*dif.T, dif)/2

            if gz <= gvz:
                flag = 1
                beta[t+1] = b
                a0 = a1 
                a1 = z
                cost[t+2] = gz
                break

        if flag == 0:
            t = t-1
            break
        else:
            delta[t+2] = (1+sqrt(1+4*delta[t+1]**2))/2

        if abs( cost[t+2] - cost[t+1])/n <= 1e-2:
            break

    a = a1 
    cost = cost/n
    return a, cost

def repmat( data, r, c ):
    return kron( ones(( r, c)), data)


class KSH(object):

    def __init__(self):
        print('Init the ksh...')
        pkl = file( '../data/data.pkl', 'rb' )
        Train, Test, self.label_index, self.sample = pickle.load(pkl)
        self.sample = array( self.sample, dtype=int) - 1
        self.label_index = array( self.label_index, dtype=int) - 1
        pkl.close()
        self.traingnd  = Train[0]
        self.traingnd = array( self.traingnd, dtype=int) - 1
        self.traindata = Train[1]
        self.testgnd   = Test[0]
        self.testgnd = array( self.testgnd, dtype=int) - 1
        self.testdata  = Test[1]

        # number of anchors
        self.m = self.sample.shape[0]
        # number of labeled training samples
        self.trn = self.label_index.shape[0]
        

    def ksh(self, bit_num):
        # kernel computing
        tn = self.testdata.shape[0]
        anchor = self.traindata[self.sample]
        KTrain = sqdist(self.traindata.T, anchor.T)
        sigma  = mean(KTrain, axis=1).mean(axis=0)
        KTrain = exp(-KTrain/(2*sigma))
        mvec   = mean(KTrain, axis=0)
        KTrain = KTrain - repmat(mvec, self.traingnd.shape[0], 1)

        #pairwise label matrix
        trngnd = mat(self.traingnd[self.label_index.T]).T
        temp   = repmat(trngnd, 1, self.trn) - repmat(trngnd.T, self.trn, 1)
        S0     = -ones( (self.trn, self.trn), float)
        tep    = (temp == 0).nonzero()
        S0[tep] = 1
        S = bit_num*S0


        # projection optimization
        KK = KTrain[self.label_index.T]
        RM = dot(KK.T,KK)
        A1 = zeros( (self.m, bit_num), float)
        flag = zeros( bit_num )
        
        for rr in range(0,bit_num):
            print rr
            if rr > 0:
                S = S- dot(y, y.T)

            LM = dot( dot(KK.T, S), KK )
            (V, U) = eigh( LM, RM, eigvals_only=False)

            
            A1[:,rr] = U[:,299]
            tep = dot(dot( A1[:,rr].T, RM ), A1[:,rr])
            A1[:,rr] = sqrt(self.trn/tep)*A1[:,rr]

            
            get_vec, cost = OptProjectionFast( KK, S, A1[:,rr], 500)
            
            y = dot(KK, A1[:,rr])
            y = ( y > 0).choose( y, 1)
            y = ( y <=0).choose( y,-1)
            y = y.reshape(y.shape[0], 1)

            y1 = dot(KK, get_vec)
            y1 = (y1>0).choose(y1,1)
            y1 = (y1<=0).choose(y1,-1)
            y1 = y1.reshape(y1.shape[0], 1)

            if dot(dot(y1.T, S), y1) > dot( dot(y.T,S), y):
                flag[rr] = 1
                A1[:,rr] = get_vec
                y = y1


        # encoding 
        Y = dot(A1.T, KTrain.T)
        Y = (Y>0).choose(Y,1)
        Y = (Y<=0).choose(Y,-1)
        

        # test 
        # encoding
        KTest = sqdist( self.testdata.T, anchor.T )
        KTest = exp( -KTest/(2*sigma) )
        KTest = KTest - repmat(mvec, tn, 1)
        tY = dot(A1.T, KTest.T)
        tY = (tY>0).choose(tY,1)
        tY = (tY<=0).choose(tY,-1)


        # search
        sim = dot(Y.T, tY)
        result_cal(sim, self.testgnd)


def result_cal(sim, testgnd):
    rank_num = [1, 5, 10, 20, 30, 50]
    findNum  = [0, 0,  0,  0,  0,  0]
    MAP      = [0, 0,  0,  0,  0,  0]
    Precision      = [0, 0,  0,  0,  0,  0]

    sim = sim.T 
    for row in sim:
        sort_d = mat(tuple(msort(row)[::-1]))
        index  = mat(tuple(row.argsort(axis=0)[::-1]))
        
        for i in range(0, testgnd.shape[0]):
            tt = -1
            for m in rank_num:
                tt += 1
                for j in range(0, m):
                    if int(testgnd[i]) == int(index[0,j]):
                        findNum[tt] += 1
            #for i in range(0, m):
            #    print testgnd[i], index[0, i]
            #    if int(testgnd[i]) == int(index[0, i]):
            #        findNum[t] += 1 

    print findNum
    for i in range(0, 6):
        MAP[i] = (findNum[i]/rank_num[i]/testgnd.shape[0])

    print MAP
        

                    




        
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
    ksh.ksh(8)
