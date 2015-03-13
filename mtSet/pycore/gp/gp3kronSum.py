import sys
sys.path.append('./../../..')
from mtSet.pycore.utils.utils import smartSum
from mtSet.pycore.mean import mean
import mtSet.pycore.covariance as covariance

import pdb
import numpy as NP
import scipy as SP
import scipy.linalg as LA
import sys
import time as TIME

from gp_base import GP

class gp3kronSum(GP):
 
    def __init__(self,mean,Cg,Cn,XX=None,S_XX=None,U_XX=None,rank=1,Xr=None,lazy=False,offset=1e-4):
        """
        Y:      Phenotype matrix
        Cg:     LIMIX trait-to-trait covariance for genetic contribution
        Cn:     LIMIX trait-to-trait covariance for noise
        XX:     Matrix for fixed sample-to-sample covariance function
        S_XX:   Eigenvalues of XX
        U_XX:   Eigenvectors of XX
        """
        # init cache
        self.cache = {} 
        # pheno
        self.setMean(mean)
        # colCovariances
        self.setColCovars(rank,Cg,Cn)
        # row covars
        self.set_XX(XX,S_XX,U_XX)
        if Xr is not None:    self.set_Xr(Xr)
        #offset for trait covariance matrices
        self.setOffset(offset)
        self.params = None
        #lazy
        self.lazy = lazy
        # time
        self.time  = {}
        self.count = {}


    def get_time(self):
        """ returns time dictionary """
        return self.time

    def get_count(self):
        """ return count dictionary """
        return self.count

    def restart(self):
        """ set all times to 0 """
        for key in self.time.keys():
            self.time[key]  = 0
            self.count[key] = 0

    def setColCovars(self,rank,Cg,Cn):
        """
        set column covariances
        """
        self.rank = rank
        # col covars
        self.Cr = covariance.lowrank(self.P,self.rank)
        self.Cr.setParams(1e-3*SP.randn(self.P*self.rank))
        self.Cg = Cg
        self.Cn = Cn

    def setMean(self,mean):
        """
        set phenotype
        """
        self.N,self.P = mean.getDimensions()
        self.mean = mean

    def setY(self,Y):
        """
        set phenotype
        """
        self.mean.setY(Y)

    def setOffset(self,offset):
        """
        set offset
        """
        self.offset = offset

    def set_XX(self,XX=None,S_XX=None,U_XX=None):
        """
        set pop struct row covariance
        """
        XXnotNone = XX is not None
        SUnotNone = S_XX is not None and U_XX is not None
        assert XXnotNone or SUnotNone, 'Specify either XX or S_XX and U_XX!'
        if SUnotNone:
            self.cache['Srstar'] = S_XX 
            self.cache['Lr'] = U_XX.T
            self.mean.setRowRotation(Lr=self.cache['Lr'])
            self.XX_has_changed = False
        else:
            self.XX = XX
            self.XX_has_changed = True

    def set_Xr(self,Xr):
        """
        set SNPs in the region
        """
        self.Xr = Xr
        self.S  = Xr.shape[1]
        self.Xr_has_changed = True

    def getParams(self):
        """
        get hper parameters
        """
        params = {}
        params['Cr'] = self.Cr.getParams()
        params['Cg'] = self.Cg.getParams()
        params['Cn'] = self.Cn.getParams()
        if 'mean' in self.params.keys():
            params['mean'] = self.mean.getParams()

        return params

    def setParams(self,params):
        """
        set hper parameters
        """
        if self.lazy:
            run_update = False
            if self.params is None:
                run_update = True
            else:
                if not(SP.allclose(self.params['Cr'],params['Cr'])):
                    run_update = True
                if not(SP.allclose(self.params['Cn'],params['Cn'])):
                    run_update = True
                if not(SP.allclose(self.params['Cg'],params['Cg'])):
                    run_update = True
        else:
            run_update = True

        if run_update:
            self.params = params
            self.updateParams()

    def updateParams(self):
        """
        update parameters
        """
        keys =self. params.keys()
        if 'Cr' in keys:
            self.Cr.setParams(self.params['Cr'])
        if 'Cg' in keys:
            self.Cg.setParams(self.params['Cg'])
        if 'Cn' in keys:
            self.Cn.setParams(self.params['Cn'])
        if 'mean' in keys:
            self.mean.setParams(self.params['mean'])

    def _update_cache(self):
        """
        Update cache
        """
        cov_params_have_changed = self.Cr.params_have_changed or self.Cg.params_have_changed or self.Cn.params_have_changed

        if self.XX_has_changed:
            start = TIME.time()
            """ Row SVD Bg + Noise """
            self.cache['Srstar'],Urstar  = LA.eigh(self.XX)
            self.cache['Lr']   = Urstar.T
            self.mean.setRowRotation(Lr=self.cache['Lr'])

            smartSum(self.time,'cache_XXchanged',TIME.time()-start)
            smartSum(self.count,'cache_XXchanged',1)
        
        if self.Xr_has_changed or self.XX_has_changed:
            start = TIME.time()
            """ rotate Xr and XrXr """
            self.cache['LXr']    = SP.dot(self.cache['Lr'],self.Xr)
            smartSum(self.time,'cache_Xrchanged',TIME.time()-start)
            smartSum(self.count,'cache_Xrchanged',1)

        if cov_params_have_changed:
            start = TIME.time()
            """ Col SVD Bg + Noise """
            S2,U2 = LA.eigh(self.Cn.K()+self.offset*SP.eye(self.P))
            self.cache['Sc2'] = S2
            US2   = SP.dot(U2,SP.diag(SP.sqrt(S2)))
            USi2  = SP.dot(U2,SP.diag(SP.sqrt(1./S2)))
            Cstar = SP.dot(USi2.T,SP.dot(self.Cg.K(),USi2))
            self.cache['Scstar'],Ucstar = LA.eigh(Cstar)
            self.cache['Lc'] = SP.dot(Ucstar.T,USi2.T)

            """ pheno """
            self.mean.setColRotation(self.cache['Lc'])

            """ region part """
            self.cache['A']   = SP.reshape(self.Cr.getParams(),(self.P,self.rank),order='F')
            self.cache['LAc'] = SP.dot(self.cache['Lc'],self.cache['A'])

        if cov_params_have_changed or self.XX_has_changed:
            """ S """
            self.cache['s'] = SP.kron(self.cache['Scstar'],self.cache['Srstar'])+1
            self.cache['d'] = 1./self.cache['s']
            self.cache['D'] = SP.reshape(self.cache['d'],(self.N,self.P), order='F')

            """ pheno """
            self.cache['LY']  = self.mean.evaluate()
            self.cache['DLY'] = self.cache['D']*self.cache['LY']

            smartSum(self.time,'cache_colSVDpRot',TIME.time()-start)
            smartSum(self.count,'cache_colSVDpRot',1)

        if cov_params_have_changed or self.XX_has_changed or self.Xr_has_changed:

            """ calculate B =  I + kron(LcA,LrXr).T*D*kron(kron(LcA,LrXr)) """
            start = TIME.time()
            W                = SP.kron(self.cache['LAc'],self.cache['LXr'])
            self.cache['DW']  = W*self.cache['d'][:,SP.newaxis]
            self.cache['DWt'] = self.cache['DW'].reshape((self.N,self.P,self.rank*self.S),order='F')
            #B  = NP.einsum('ijk,jl->ilk',self.cache['DWt'],self.cache['LAc'])
            #B  = NP.einsum('ji,jlk->ilk',self.cache['LXr'],B)
            B = SP.tensordot(self.cache['DWt'],self.cache['LAc'],axes=(1,0)) 
            B = NP.transpose(B, (0, 2, 1))
            B = SP.tensordot(self.cache['LXr'],B,axes=(0,0))
            B = B.reshape((self.rank*self.S,self.rank*self.S),order='F')
            B+= SP.eye(self.rank*self.S)
            smartSum(self.time,'cache_calcB',TIME.time()-start)
            smartSum(self.count,'cache_calcB',1)

            """ invert B """
            start = TIME.time()
            self.cache['cholB'] = LA.cholesky(B).T
            self.cache['Bi']    = LA.cho_solve((self.cache['cholB'],True),SP.eye(self.S*self.rank))
            smartSum(self.time,'cache_invB',TIME.time()-start)
            smartSum(self.count,'cache_invB',1)
            
            """ pheno """
            start = TIME.time()
            Z = SP.dot(self.cache['LXr'].T,SP.dot(self.cache['DLY'],self.cache['LAc']))
            self.cache['z']           = SP.reshape(Z,(self.S*self.rank), order='F')
            self.cache['Biz']         = LA.cho_solve((self.cache['cholB'],True),self.cache['z'])
            BiZ = SP.reshape(self.cache['Biz'],(self.S,self.rank), order='F')
            self.cache['DLYpDLXBiz']  = SP.dot(self.cache['LXr'],SP.dot(BiZ,self.cache['LAc'].T))
            self.cache['DLYpDLXBiz'] *= -self.cache['D']
            self.cache['DLYpDLXBiz'] += self.cache['DLY']
            smartSum(self.time,'cache_phenoCalc',TIME.time()-start)
            smartSum(self.count,'cache_phenoCalc',1)

        self.XX_has_changed = False
        self.Xr_has_changed = False
        self.Y_has_changed  = False
        self.Cr.params_have_changed = False
        self.Cg.params_have_changed = False
        self.Cn.params_have_changed = False


    def LML(self,params=None,*kw_args):
        """
        calculate LML
        """
        if params is not None:
            self.setParams(params)

        self._update_cache()
        
        start = TIME.time()

        #1. const term
        lml  = self.N*self.P*SP.log(2*SP.pi)

        #2. logdet term
        lml += SP.sum(SP.log(self.cache['Sc2']))*self.N + SP.log(self.cache['s']).sum()
        lml += 2*SP.log(SP.diag(self.cache['cholB'])).sum()

        #3. quatratic term
        lml += (self.cache['LY']*self.cache['DLY']).sum()
        lml -= SP.dot(self.cache['z'],self.cache['Biz'])

        lml *= 0.5

        smartSum(self.time,'lml',TIME.time()-start)
        smartSum(self.count,'lml',1)


        return lml


    def LMLdebug(self):
        """
        LML function for debug
        """
        assert self.N*self.P<2000, 'gp3kronSum:: N*P>=2000'

        Rr = SP.dot(self.Xr,self.Xr.T)
        y  = SP.reshape(self.Y,(self.N*self.P), order='F') 

        K  = SP.kron(self.Cr.K(),Rr)
        K += SP.kron(self.Cg.K(),self.XX)
        K += SP.kron(self.Cn.K(),SP.eye(self.N))

        cholK = LA.cholesky(K)
        Kiy   = LA.cho_solve((cholK,False),y)

        lml  = y.shape[0]*SP.log(2*SP.pi)
        lml += 2*SP.log(SP.diag(cholK)).sum()
        lml += SP.dot(y,Kiy)
        lml *= 0.5

        return lml


    def LMLgrad(self,params=None,**kw_args):
        """
        LML gradient
        """
        if params is not None:
            self.setParams(params)
        self._update_cache()
        RV = {}
        covars = ['Cr','Cg','Cn']
        for covar in covars:        
            RV[covar] = self._LMLgrad_covar(covar)
        if 'mean' in self.params.keys():
            RV['mean'] = self._LMLgrad_mean()
        return RV


    def _LMLgrad_covar(self,covar,**kw_args):
        """
        calculates LMLgrad for covariance parameters
        """
        start = TIME.time()

        # precompute some stuff
        if covar=='Cr':     n_params = self.Cr.getNumberParams()
        elif covar=='Cg':   n_params = self.Cg.getNumberParams()
        elif covar=='Cn':   n_params = self.Cn.getNumberParams()

        KDW  = SP.zeros_like(self.cache['DW'])  
        if covar=='Cr':
            #_KDWt = NP.einsum('ij,ilk->jlk',self.cache['LXr'],self.cache['DWt'])
            #_KDWt = NP.einsum('ij,jlk->ilk',self.cache['LXr'],_KDWt)
            _KDWt = NP.tensordot(self.cache['LXr'],self.cache['DWt'],axes=(0,0))
            _KDWt = NP.tensordot(self.cache['LXr'],_KDWt,axes=(1,0))
            _KDLYpDLXBiz = SP.dot(self.cache['LXr'].T,self.cache['DLYpDLXBiz'])
            _KDLYpDLXBiz = SP.dot(self.cache['LXr'],_KDLYpDLXBiz)
            LRLdiag = (self.cache['LXr']**2).sum(1)
        elif covar=='Cg':
            _KDWt = self.cache['Srstar'][:,SP.newaxis,SP.newaxis]*self.cache['DWt']
            _KDLYpDLXBiz = self.cache['Srstar'][:,SP.newaxis]*self.cache['DLYpDLXBiz']
            LRLdiag = self.cache['Srstar']
        else:
            _KDWt = self.cache['DWt']
            _KDLYpDLXBiz = self.cache['DLYpDLXBiz']
            LRLdiag = SP.ones(self.N)

        smartSum(self.time,'lmlgrad_trace2_rKDW_%s'%covar,TIME.time()-start)
        smartSum(self.count,'lmlgrad_trace2_rKDW_%s'%covar,1)

        # fill gradient vector
        RV = SP.zeros(n_params)
        for i in range(n_params):
            #0. calc LCL

            if covar=='Cr':     C = self.Cr.Kgrad_param(i)
            elif covar=='Cg':   C = self.Cg.Kgrad_param(i)
            elif covar=='Cn':   C = self.Cn.Kgrad_param(i)

            LCL = SP.dot(self.cache['Lc'],SP.dot(C,self.cache['Lc'].T))

            #1. der of log det
            start = TIME.time()
            kronDiag  = SP.kron(LCL.diagonal(),LRLdiag)
            RV[i] = SP.dot(self.cache['d'],kronDiag)
            smartSum(self.time,'lmlgrad_trace1_%s'%covar,TIME.time()-start)
            smartSum(self.count,'lmlgrad_trace1_%s'%covar,1)
            start = TIME.time()
            #KDWt  = NP.einsum('ijk,jl->ilk',_KDWt,LCL)
            KDWt = NP.tensordot(_KDWt,LCL,axes=(1,0))
            smartSum(self.time,'lmlgrad_trace2_cKDW_%s'%covar,TIME.time()-start)
            smartSum(self.count,'lmlgrad_trace2_cKDW_%s'%covar,1)
            
            start = TIME.time()
            #DKDWt = NP.einsum('ij,ijk->ijk',self.cache['D'],KDWt)
            #WDKDWt = NP.einsum('ijk,jl->ilk',DKDWt, self.cache['LAc'])
            #WDKDWt = NP.einsum('ij,ilk->jlk',self.cache['LXr'],WDKDWt)
            DKDWt = self.cache['D'][:,SP.newaxis,:]*KDWt
            WDKDWt = NP.tensordot(DKDWt,self.cache['LAc'],axes=(2,0))
            WDKDWt = NP.tensordot(self.cache['LXr'],WDKDWt,axes=(0,0))
            WDKDWt = NP.transpose(WDKDWt,(0,2,1))
            WDKDW  = WDKDWt.reshape((self.rank*self.S,self.rank*self.S),order='F')
            smartSum(self.time,'lmlgrad_trace2_WDKDW_%s'%covar,TIME.time()-start)
            smartSum(self.count,'lmlgrad_trace2_WDKDW_%s'%covar,1)

            RV[i] -= (WDKDW*self.cache['Bi']).sum() 

            #2. der of quad form
            start = TIME.time()
            KDLYpDLXBiz = SP.dot(_KDLYpDLXBiz,LCL.T)
            RV[i] -= (self.cache['DLYpDLXBiz']*KDLYpDLXBiz).sum()
            smartSum(self.time,'lmlgrad_quadForm_%s'%covar,TIME.time()-start)
            smartSum(self.count,'lmlgrad_quadForm_%s'%covar,1)

            RV[i] *= 0.5

        return RV

    def _LMLgrad_mean(self):
        """ LMLgradient with respect to the mean params """
        n_params = self.params['mean'].shape[0]
        RV = SP.zeros(n_params)
        for i in range(n_params):
            dF = self.mean.getGradient(i)
            RV[i] = (dF*self.cache['DLYpDLXBiz']).sum()
        return RV

    def _LMLgrad_covar_debug(self,covar):

        assert self.N*self.P<2000, 'gp3kronSum:: N*P>=2000'

        Rr = SP.dot(self.Xr,self.Xr.T)
        y  = SP.reshape(self.Y,(self.N*self.P), order='F') 

        K  = SP.kron(self.Cr.K(),Rr)
        K += SP.kron(self.Cg.K(),self.XX)
        K += SP.kron(self.Cn.K(),SP.eye(self.N))

        cholK = LA.cholesky(K).T
        Ki    = LA.cho_solve((cholK,True),SP.eye(y.shape[0]))
        Kiy   = LA.cho_solve((cholK,True),y)

        if covar=='Cr':     n_params = self.Cr.getNumberParams()
        elif covar=='Cg':   n_params = self.Cg.getNumberParams()
        elif covar=='Cn':   n_params = self.Cn.getNumberParams()

        RV = SP.zeros(n_params)

        for i in range(n_params):
            #0. calc grad_i
            if covar=='Cr':
                C   = self.Cr.Kgrad_param(i)
                Kgrad  = SP.kron(C,Rr)
            elif covar=='Cg':
                C   = self.Cg.Kgrad_param(i)
                Kgrad  = SP.kron(C,self.XX)
            elif covar=='Cn':
                C   = self.Cn.Kgrad_param(i)
                Kgrad  = SP.kron(C,SP.eye(self.N))

            #1. der of log det
            RV[i]  = 0.5*(Ki*Kgrad).sum()
            
            #2. der of quad form
            RV[i] -= 0.5*(Kiy*SP.dot(Kgrad,Kiy)).sum()

        return RV

    def predict(self,terms=None):
        if terms is None:
            terms = ['Cr','Cg']

        self._update_cache()
        Kiy = SP.dot(self.cache['Lr'].T,SP.dot(self.cache['DLYpDLXBiz'],self.cache['Lc']))

        RV = SP.zeros((self.N,self.P))
        for term_i in terms: 
            if term_i=='Cr':
                C = self.Cr.K()
                RKiy = SP.dot(self.Xr,SP.dot(self.Xr.T,Kiy))
                RV += SP.dot(RKiy,C)
            elif term_i=='Cg':
                C = self.Cg.K()
                RKiy = SP.dot(self.XX,Kiy)
                RV += SP.dot(RKiy,C)
            elif term_i=='Cn':
                C = self.Cn.K()
                RV += SP.dot(Kiy,C)

        return RV

    def simulate(self,standardize=True):
        self._update_cache()
        RV = SP.zeros((self.N,self.P))
        # region
        Z = SP.randn(self.S,self.P)
        Sc,Uc = LA.eigh(self.Cr.K())
        Sc[Sc<1e-9] = 0
        USh_c = Uc*Sc[SP.newaxis,:]**0.5 
        RV += SP.dot(SP.dot(self.Xr,Z),USh_c.T)
        # background
        Z = SP.randn(self.N,self.P)
        USh_r = self.cache['Lr'].T*self.cache['Srstar'][SP.newaxis,:]**0.5
        Sc,Uc = LA.eigh(self.Cg.K())
        Sc[Sc<1e-9] = 0
        USh_c = Uc*Sc[SP.newaxis,:]**0.5
        RV += SP.dot(SP.dot(USh_r,Z),USh_c.T)
        # noise
        Z = SP.randn(self.N,self.P)
        Sc,Uc = LA.eigh(self.Cn.K())
        Sc[Sc<1e-9] = 0
        USh_c = Uc*Sc[SP.newaxis,:]**0.5 
        RV += SP.dot(Z,USh_c.T)
        # standardize
        if standardize:
            RV-=RV.mean(0)
            RV/=RV.std(0) 
        return RV

    def getPosteriorFactorWeights(self,debug=False):
        """
        get posterior weights on low-rank genetic factors 
        """
        self._update_cache()

        F = self.cache['A'].shape[1] * self.Xr.shape[1]
        W   = SP.kron(self.cache['LAc'],self.cache['LXr'])
        Sigma = LA.inv(SP.eye(F) + SP.dot(W.T,self.cache['DW']))
        mean  = SP.dot(Sigma,self.cache['z'])

        if debug:
            assert self.N*self.P<=2000, 'N*P>2000!'
            Cr = self.Cr.K()
            Cn = self.Cn.K()
            Cg = self.Cg.K()
            y  = SP.reshape(self.Y,(self.N*self.P), order='F')
            _Sigma = LA.inv(SP.eye(F) + SP.dot(SP.kron(self.cache['A'].T,self.Xr.T),LA.solve(SP.kron(Cg,self.XX) + SP.kron(Cn,SP.eye(self.N)),SP.kron(self.cache['A'],self.Xr))))
            _mean  = SP.dot(Sigma,SP.dot(SP.kron(self.cache['A'].T,self.Xr.T),LA.solve(SP.kron(Cg,self.XX) + SP.kron(Cn,SP.eye(self.N)),y)))

            assert SP.allclose(_Sigma,Sigma,rtol=1e-3,atol=1e-5), 'ouch'
            assert SP.allclose(_mean,mean,rtol=1e-3,atol=1e-5), 'ouch'

        return mean,Sigma

    def getPosteriorSnpWeights(self,matrix=False):
        """
        get posterior on the number of Snps
        """
        meanF, SigmaF = self.getPosteriorFactorWeights()
        V = SP.kron(self.cache['A'],SP.eye(self.Xr.shape[1]))
        mean = SP.dot(V,meanF)
        Sigma = SP.dot(V,SP.dot(SigmaF,V.T))
        if matrix:
            M = SP.reshape(mean,(self.S,self.P),order='F')
            S = SP.sqrt(Sigma.diagonal()).reshape((self.S,self.P),order='F')
            return mean, Sigma, M, S
        else:
            return mean, Sigma
        
        
                
    

if 0:

    def LMLgrad(self,hyperparams,**kw_args):
        """
        evaludates the gradient of the log marginal likelihood
        Input:
        hyperparams: dictionary
        priors:      prior beliefs for the hyperparameter
        """
        self._update_inputs(hyperparams)
        RV = {}
        # gradient with respect to hyperparameters
        RV.update(self._LMLgrad_covar(hyperparams))
        # gradient with respect to noise parameters
        if self.likelihood is not None:
            RV.update(self._LMLgrad_lik(hyperparams))
        # gradient with respect to X
        RV.update(self._LMLgrad_x(hyperparams))
        return RV

    def _LMLgrad_x(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with
        respect to the latent variables
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LML_grad_x')
            return {'X': SP.zeros(hyperparams['X'].shape)}
        except LA.LinAlgError:
            LG.error('linalg exception in _LML_grad_x')
            return {'X': SP.zeros(hyperparams['X'].shape)}
  
        W = KV['W']
        LMLgrad = SP.zeros((self.n,self.d))
        for d in xrange(self.d):
            Kd_grad = self.covar.Kgrad_x(hyperparams['covar'],self.X,None,d)
            LMLgrad[:,d] = SP.sum(W*Kd_grad,axis=0)

        if self.debugging:
            # compare to explicit solution
            LMLgrad2 = SP.zeros((self.n,self.d))
            for n in xrange(self.n):
                for d in xrange(self.d):
                    Knd_grad = self.covar.Kgrad_x(hyperparams['covar'],self.X,n,d)
                    LMLgrad2[n,d] = 0.5*(W*Knd_grad).sum()
            assert SP.allclose(LMLgrad,LMLgrad2), 'ouch, something is wrong'
        
        return {'X':LMLgrad}

    def _update_inputs(self,hyperparams):
        """ update the inputs from gplvm model """
        if 'X' in hyperparams:
            self.X = hyperparams['X']
        
