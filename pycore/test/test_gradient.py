import sys
sys.path.append('./../..')
sys.path.append('/Users/casale/Documents/limix/limix/build/release.darwin/interfaces/python')
import limix

import ipdb
import scipy as SP
import scipy.linalg as LA 
import time as TIME
import copy

import pycore.mean as MEAN
from pycore.gp import gp2kronSum
from pycore.gp import gp2kronSumSvd
from pycore.gp import gp3kronSum
import pycore.optimize.optimize_bfgs as OPT


if __name__ == "__main__":

    # generate data
    h2 = 0.3
    N = 1000; P = 4; S = 1000
    X = 1.*(SP.rand(N,S)<0.2)
    beta = SP.randn(S,P)
    Yg = SP.dot(X,beta); Yg*=SP.sqrt(h2/Yg.var(0).mean())
    Yn = SP.randn(N,P); Yn*=SP.sqrt((1-h2)/Yn.var(0).mean())
    Y  = Yg+Yn; Y-=Y.mean(0); Y/=Y.std(0)
    XX = SP.dot(X,X.T)
    XX/= XX.diagonal().mean()
    Xr = 1.*(SP.rand(N,10)<0.2)
    Xr*= SP.sqrt(N/(Xr**2).sum())

    # define mean term
    mean = MEAN.mean(Y)
    # add first fixed effect
    F = 1.*(SP.rand(N,2)<0.2); A = SP.eye(P)
    mean.addFixedEffect(F=F,A=A)
    # add first fixed effect
    F = 1.*(SP.rand(N,3)<0.2); A = SP.ones((1,P))
    mean.addFixedEffect(F=F,A=A)

    # define covariance matrices
    Cg = limix.CFreeFormCF(P)
    Cn = limix.CFreeFormCF(P)
    
    if 1:
        # generate parameters
        params = {}
        params['Cg']   = SP.randn(int(0.5*P*(P+1)))
        params['Cn']   = SP.randn(int(0.5*P*(P+1)))
        params['mean'] = 1e-2*SP.randn(mean.getParams().shape[0])
        print "check gradient with gp2kronSum"
        gp = gp2kronSum(mean,Cg,Cn,XX)
        gp.setParams(params)
        gp.checkGradient()
        print "test optimization"
        conv,info = OPT.opt_hyper(gp,params,factr=1e3)
        print conv
        ipdb.set_trace()

    if 1:
        # generate parameters
        params = {}
        params['Cr']   = SP.randn(P)
        params['Cg']   = SP.randn(int(0.5*P*(P+1)))
        params['Cn']   = SP.randn(int(0.5*P*(P+1)))
        params['mean'] = 1e-2*SP.randn(mean.getParams().shape[0])
        print "check gradient with gp3kronSum"
        gp = gp3kronSum(mean,Cg,Cn,XX,Xr=Xr)
        gp.setParams(params)
        gp.checkGradient()
        print "test optimization"
        conv,info = OPT.opt_hyper(gp,params,factr=1e3)
        print conv
        ipdb.set_trace()

    if 1:
        # generate parameters
        params = {}
        params['Cr']   = SP.randn(P)
        params['Cn']   = SP.randn(int(0.5*P*(P+1)))
        params['mean'] = 1e-2*SP.randn(mean.getParams().shape[0])
        print "check gradient with gp2kronSumSVD"
        gp = gp2kronSumSvd(mean,Cn,Xr=Xr)
        gp.setParams(params)
        gp.checkGradient()
        print "test optimization"
        conv,info = OPT.opt_hyper(gp,params,factr=1e3)
        print conv
        ipdb.set_trace()

    if 1:
        # test convergence for gp2kronSum with mask on Cg
        params = {}
        params['Cg']   = SP.zeros(int(0.5*P*(P+1)))
        params['Cn']   = SP.randn(int(0.5*P*(P+1)))
        params['mean'] = 1e-2*SP.randn(mean.getParams().shape[0])
        Ifilter = {'Cg':SP.ones(int(0.5*P*(P+1)),dtype=bool),
                    'Cn':SP.ones(int(0.5*P*(P+1)),dtype=bool),
                    'mean':SP.ones(params['mean'].shape[0],dtype=bool)} 
        gp = gp2kronSum(mean,Cg,Cn,XX)
        print "test optimization"
        conv,info = OPT.opt_hyper(gp,params,Ifilter=Ifilter,factr=1e3)
        print conv
        ipdb.set_trace()



