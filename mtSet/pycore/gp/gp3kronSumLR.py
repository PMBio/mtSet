import sys
sys.path.append('./../../..')
from mtSet.pycore.utils.utils import smartSum
import mtSet.pycore.covariance as covariance

import pdb
import numpy as NP
import scipy as SP
import scipy.linalg as LA
import sys
import time as TIME

from gp_base import GP

class gp3kronSumLR(GP):
 
    def __init__(self,Y,Cg,Cn,X,rank=1,Xr=None,lazy=False,offset=1e-4):
        """
        Y:      Phenotype matrix
        Cg:     LIMIX trait-to-trait covariance for genetic contribution
        Cn:     LIMIX trait-to-trait covariance for noise
        """
        # init cache
        self.cache = {} 
        # pheno
        self.setY(Y)
        # colCovariances
        self.setColCovars(rank,Cg,Cn)
        # row covars
        self.set_X(X)
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

    def setY(self,Y):
        """
        set phenotype
        """
        self.N,self.P = Y.shape 
        self.Y = Y
        self.Y_has_changed = True

    def setOffset(self,offset):
        """
        set offset
        """
        self.offset = offset

    def set_X(self,X):
        """
        set pop struct row covariance
        """
        self.X = X
        self.X_has_changed = True

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

    def _update_cache(self):
        """
        Update cache
        """
        cov_params_have_changed = self.Cr.params_have_changed or self.Cg.params_have_changed or self.Cn.params_have_changed

        if self.X_has_changed:
            self.cache['trXX'] = SP.sum(self.X**2)
            self.cache['XX'] = SP.dot(self.X.T,self.X)
            self.cache['XY'] = SP.dot(self.X.T,self.Y)
            self.cache['XXY'] = SP.dot(self.X,self.cache['XY'])
            self.cache['XXXY'] = SP.dot(self.cache['XX'],self.cache['XY'])
            self.cache['XXXX'] = SP.dot(self.cache['XX'],self.cache['XX'])

        if self.Xr_has_changed:
            self.cache['trXrXr'] = SP.sum(self.Xr**2)
            self.cache['XrXr'] = SP.dot(self.Xr.T,self.Xr)
            self.cache['XrY'] = SP.dot(self.Xr.T,self.Y)
            self.cache['XrXrY'] = SP.dot(self.Xr,self.cache['XrY'])
            self.cache['XrXrXrY'] = SP.dot(self.cache['XrXr'],self.cache['XrY'])
            self.cache['XrXrXrXr'] = SP.dot(self.cache['XrXr'],self.cache['XrXr'])

        if self.X_has_changed or self.Xr_has_changed:
            self.cache['XXr'] = SP.dot(self.X.T,self.Xr)
            self.cache['XXrXrY'] = SP.dot(self.cache['XXr'],self.cache['XrY'])
            self.cache['XXrXrX'] = SP.dot(self.cache['XXr'],self.cache['XXr'].T)
            self.cache['XrXXY'] = SP.dot(self.cache['XXr'].T,self.cache['XY'])
            self.cache['XrXXXr'] = SP.dot(self.cache['XXr'].T,self.cache['XXr'])
            self.cache['XrXrXrX'] = SP.dot(self.cache['XrXr'],self.cache['XXr'].T)
            self.cache['XrXXX'] = SP.dot(self.cache['XXr'].T,self.cache['XX'])
        
        if cov_params_have_changed:
            start = TIME.time()
            """ Col SVD Bg + Noise """
            S2,U2 = LA.eigh(self.Cn.K()+self.offset*SP.eye(self.P))
            self.cache['Sc2'] = S2
            US2   = SP.dot(U2,SP.diag(SP.sqrt(S2)))
            USi2  = SP.dot(U2,SP.diag(SP.sqrt(1./S2)))
            self.cache['Lc'] = USi2.T
            self.cache['Cstar'] = SP.dot(USi2.T,SP.dot(self.Cg.K(),USi2))
            self.cache['Scstar'],Ucstar = LA.eigh(self.cache['Cstar'])
            self.cache['CstarH'] = Ucstar*((self.cache['Scstar']**(0.5))[SP.newaxis,:])
            E = SP.reshape(self.Cr.getParams(),(self.P,self.rank),order='F')
            self.cache['Estar'] = SP.dot(USi2.T,E)
            self.cache['CE'] = SP.dot(self.cache['CstarH'].T,self.cache['Estar'])
            self.cache['EE'] = SP.dot(self.cache['Estar'].T,self.cache['Estar'])

        if cov_params_have_changed or self.Y_has_changed:
            self.cache['LY'] = SP.dot(self.Y,self.cache['Lc'].T)
            
        if cov_params_have_changed or self.Xr_has_changed or self.Y_has_changed:
            self.cache['XrLY'] = SP.dot(self.cache['XrY'],self.cache['Lc'].T)
            self.cache['WLY1'] = SP.dot(self.cache['XrLY'],self.cache['Estar'])
            self.cache['XrXrLY'] = SP.dot(self.cache['XrXrY'],self.cache['Lc'].T)
            self.cache['XrXrXrLY'] = SP.dot(self.cache['XrXrXrY'],self.cache['Lc'].T)

        if cov_params_have_changed or self.X_has_changed or self.Y_has_changed:
            self.cache['XLY'] = SP.dot(self.cache['XY'],self.cache['Lc'].T)
            self.cache['WLY2'] = SP.dot(self.cache['XLY'],self.cache['CstarH'])
            self.cache['XXLY'] = SP.dot(self.cache['XXY'],self.cache['Lc'].T)
            self.cache['XXXLY'] = SP.dot(self.cache['XXXY'],self.cache['Lc'].T)

        if cov_params_have_changed or self.X_has_changed or self.Xr_has_changed:
            """ calculate B """
            B11 = SP.kron(self.cache['EE'],self.cache['XrXr'])
            B11+= SP.eye(B11.shape[0])
            B21 = SP.kron(self.cache['CE'],self.cache['XXr'])
            B22 = SP.kron(SP.diag(self.cache['Scstar']),self.cache['XX'])
            B22+= SP.eye(B22.shape[0])
            B = SP.bmat([[B11,B21.T],[B21,B22]])
            self.cache['cholB'] = LA.cholesky(B).T 
            self.cache['Bi']    = LA.cho_solve((self.cache['cholB'],True),SP.eye(B.shape[0]))

        if cov_params_have_changed or self.X_has_changed or self.Xr_has_changed or self.Y_has_changed:
            self.cache['WLY'] = SP.concatenate([SP.reshape(self.cache['WLY1'],(self.cache['WLY1'].size,1),order='F'),
                                                SP.reshape(self.cache['WLY2'],(self.cache['WLY2'].size,1),order='F')])
            self.cache['BiWLY'] = SP.dot(self.cache['Bi'],self.cache['WLY'])
            self.cache['XXrXrLY'] = SP.dot(self.cache['XXrXrY'],self.cache['Lc'].T)
            self.cache['XrXXLY'] = SP.dot(self.cache['XrXXY'],self.cache['Lc'].T)

        self.Xr_has_changed = False
        self.X_has_changed = False
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
        lml += SP.sum(SP.log(self.cache['Sc2']))*self.N
        lml += 2*SP.log(SP.diag(self.cache['cholB'])).sum()

        #3. quatratic term
        lml += SP.sum(self.cache['LY']*self.cache['LY'])
        lml -= SP.sum(self.cache['WLY']*self.cache['BiWLY'])

        lml *= 0.5

        smartSum(self.time,'lml',TIME.time()-start)
        smartSum(self.count,'lml',1)

        return lml


    def LMLdebug(self):
        """
        LML function for debug
        """
        assert self.N*self.P<5000, 'gp3kronSum:: N*P>=5000'

        Rr = SP.dot(self.Xr,self.Xr.T)
        XX = SP.dot(self.X,self.X.T)
        y  = SP.reshape(self.Y,(self.N*self.P), order='F') 

        K  = SP.kron(self.Cr.K(),Rr)
        K += SP.kron(self.Cg.K(),XX)
        K += SP.kron(self.Cn.K()+1e-4*SP.eye(self.P),SP.eye(self.N))

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

        if covar=='Cr':
            trR = self.cache['trXrXr']
            RY = self.cache['XrXrY'] 
            RLY = self.cache['XrXrLY'] 
            WrRY1 = self.cache['XrXrXrY'] 
            WrRY2 = self.cache['XXrXrY'] 
            WrRLY1 = self.cache['XrXrXrLY'] 
            WrRLY2 = self.cache['XXrXrLY'] 
            XrRXr = self.cache['XrXrXrXr']
            XrRX = self.cache['XrXrXrX']
            XRX = self.cache['XXrXrX']
        elif covar=='Cg':
            trR = self.cache['trXX']
            RY = self.cache['XXY'] 
            RLY = self.cache['XXLY'] 
            WrRY1 = self.cache['XrXXY'] 
            WrRY2 = self.cache['XXXY'] 
            WrRLY1 = self.cache['XrXXLY'] 
            WrRLY2 = self.cache['XXXLY'] 
            XrRXr = self.cache['XrXXXr']
            XrRX = self.cache['XrXXX']
            XRX = self.cache['XXXX']
        else:
            trR = self.N
            RY = self.Y
            RLY = self.cache['LY'] 
            WrRY1 = self.cache['XrY']
            WrRY2 = self.cache['XY']
            WrRLY1 = self.cache['XrLY']
            WrRLY2 = self.cache['XLY']
            XrRXr = self.cache['XrXr']
            XrRX = self.cache['XXr'].T
            XRX = self.cache['XX']

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

            ELCL = SP.dot(self.cache['Estar'].T,LCL)
            ELCLE = SP.dot(ELCL,self.cache['Estar'])
            ELCLCsh = SP.dot(ELCL,self.cache['CstarH'])
            CshLCL = SP.dot(self.cache['CstarH'].T,LCL) 
            CshLCLCsh = SP.dot(CshLCL,self.cache['CstarH'])

            # WCoRW
            WCoRW11 = SP.kron(ELCLE,XrRXr)
            WCoRW12 = SP.kron(ELCLCsh,XrRX)
            WCoRW22 = SP.kron(CshLCLCsh,XRX)
            WCoRW = SP.array(SP.bmat([[WCoRW11,WCoRW12],[WCoRW12.T,WCoRW22]]))
            # WCoRLY
            WCoRLY1 = SP.dot(WrRLY1,ELCL.T) 
            WCoRLY2 = SP.dot(WrRLY2,CshLCL.T)
            WCoRLY = SP.concatenate([SP.reshape(WCoRLY1,(WCoRLY1.size,1),order='F'),
                                     SP.reshape(WCoRLY2,(WCoRLY2.size,1),order='F')]) 
            # CoRLY
            CoRLY = SP.dot(RLY,LCL.T)

            #1. der of log det
            start = TIME.time()
            trC = LCL.diagonal().sum()
            RV[i] = trC*trR
            RV[i]-= SP.sum(self.cache['Bi']*WCoRW)
            smartSum(self.time,'lmlgrad_trace2_WDKDW_%s'%covar,TIME.time()-start)
            smartSum(self.count,'lmlgrad_trace2_WDKDW_%s'%covar,1)

            #2. der of quad form
            start = TIME.time()
            RV[i] -= SP.sum(self.cache['LY']*CoRLY)
            RV[i] -= SP.sum(self.cache['BiWLY']*SP.dot(WCoRW,self.cache['BiWLY'])) 
            RV[i] += 2*SP.sum(self.cache['BiWLY']*WCoRLY) 
            
            smartSum(self.time,'lmlgrad_quadForm_%s'%covar,TIME.time()-start)
            smartSum(self.count,'lmlgrad_quadForm_%s'%covar,1)

            RV[i] *= 0.5

        return RV

    def LMLgrad_debug(self,params=None,**kw_args):
        """
        LML gradient
        """
        if params is not None:
            self.setParams(params)
        RV = {}
        covars = ['Cr','Cg','Cn']
        for covar in covars:
            RV[covar] = self._LMLgrad_covar_debug(covar)
        return RV

    def _LMLgrad_covar_debug(self,covar):

        assert self.N*self.P<5000, 'gp3kronSum:: N*P>=5000'

        Rr = SP.dot(self.Xr,self.Xr.T)
        XX = SP.dot(self.X,self.X.T)
        y  = SP.reshape(self.Y,(self.N*self.P), order='F') 

        K  = SP.kron(self.Cr.K(),Rr)
        K += SP.kron(self.Cg.K(),XX)
        K += SP.kron(self.Cn.K()+1e-4*SP.eye(self.P),SP.eye(self.N))

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
                Kgrad  = SP.kron(C,XX)
            elif covar=='Cn':
                C   = self.Cn.Kgrad_param(i)
                Kgrad  = SP.kron(C,SP.eye(self.N))

            #1. der of log det
            RV[i]  = 0.5*(Ki*Kgrad).sum()
            
            #2. der of quad form
            RV[i] -= 0.5*(Kiy*SP.dot(Kgrad,Kiy)).sum()

        return RV


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
        
