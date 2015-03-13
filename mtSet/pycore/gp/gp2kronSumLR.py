import sys
sys.path.append('./../../..')
from mtSet.pycore.utils.utils import smartSum
from mtSet.pycore.mean import mean
import mtSet.pycore.covariance as covariance

import pdb
import numpy as NP
import scipy as SP
import scipy.linalg as LA
import numpy.linalg as NLA
import sys
import time as TIME
import warnings

from gp_base import GP

class gp2kronSumLR(GP):
 
    def __init__(self,Y,Cn,F=None,rank=1,Xr=None,offset=1e-4,tol=1e-9):
        """
        Y:      Phenotype matrix
        Cn:     LIMIX trait-to-trait covariance for noise
        rank:   rank of the region term
        Xr:     Region term NxS (Remark: fast inference requires S<<N)
        """
        # pheno
        self.setY(Y)
        # fixed effects
        self.setFixedEffect(F)
        # colCovariances
        self.setColCovars(rank,Cn)
        # set tol
        self.tol = tol
        # row covars
        if Xr is not None:    self.set_Xr(Xr)
        #offset for trait covariance matrices
        self.setOffset(offset)
        self.params = None
        # time
        self.time = {}
        self.count = {}
        #cache init
        self.cache = {}

        self.debug = False

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

    def setFixedEffect(self,F=None):
        """ set fixed effect sample design """
        self.F = F
        if self.F is not None:
            self.K = self.F.shape[1]
        self.F_has_changed = True

    def setColCovars(self,rank,Cn):
        """
        set column covariances
        """
        self.rank=rank
        # col covars
        self.Cr = covariance.lowrank(self.P,self.rank)
        self.Cr.setParams(1e-3*SP.randn(self.P*self.rank))
        self.Cn = Cn

    def setY(self,Y):
        """
        set pheno
        """
        self.N, self.P = Y.shape
        self.Y = Y
        self.Y_has_changed = True

    def setOffset(self,offset):
        """
        set offset
        """
        self.offset = offset

    def set_Xr(self,Xr):
        """
        set SNPs in the region
        """
        self.Xr = Xr
        self.S = self.Xr.shape[1]
        self.Xr_has_changed = True

    def getParams(self):
        """
        get hper parameters
        """
        params = {}
        params['Cr'] = self.Cr.getParams()
        params['Cn'] = self.Cn.getParams()

        return params

    def setParams(self,params):
        """
        set hper parameters
        """
        self.params = params
        self.updateParams()

    def updateParams(self):
        """
        update parameters
        """
        keys =self. params.keys()
        if 'Cr' in keys:
            self.Cr.setParams(self.params['Cr'])
        if 'Cn' in keys:
            self.Cn.setParams(self.params['Cn'])

    def _update_cache(self):
        """
        Update cache
        """
        cov_params_have_changed = self.Cr.params_have_changed or self.Cn.params_have_changed

        if self.Xr_has_changed:
            start = TIME.time()
            """ Row SVD on small matrix """
            Ug,Sgh,Vg = NLA.svd(self.Xr,full_matrices=0)
            I = Sgh<self.tol
            if I.any():
                warnings.warn('Xr has dependent columns, dimensionality reduced')
                Sgh = Sgh[~I]
                Ug = Ug[:,~I]
                Vg = SP.eye(Sgh.shape[0])
                Xr = Ug*Sgh[SP.newaxis,:]
                self.set_Xr(Xr)
            self.cache['Sg'] = Sgh**2
            self.cache['Wr'] = Ug.T
            self.cache['Vg'] = Vg
            self.cache['trXrXr'] = self.cache['Sg'].sum()

        if cov_params_have_changed:
            start = TIME.time()
            """ Col SVD on big matrix """
            self.cache['Sn'],Un = LA.eigh(self.Cn.K()+self.offset*SP.eye(self.P))
            self.cache['Lc'] = (self.cache['Sn']**(-0.5))[:,SP.newaxis]*Un.T
            E = SP.reshape(self.Cr.getParams(),(self.P,self.rank),order='F')
            Estar = SP.dot(self.cache['Lc'],E)
            Ue,Seh,Ve = NLA.svd(Estar,full_matrices=0)
            self.cache['Se'] = Seh**2
            self.cache['Wc'] = Ue.T

        if cov_params_have_changed or self.Xr_has_changed:
            """ S """
            self.cache['s'] = SP.kron(1./self.cache['Se'],1./self.cache['Sg'])+1
            self.cache['d'] = 1./self.cache['s']
            self.cache['D'] = SP.reshape(self.cache['d'],(self.S,self.rank), order='F')

        if self.Xr_has_changed or self.Y_has_changed:
            """ phenos transf """
            self.cache['WrLrY'] = SP.dot(self.cache['Wr'],self.Y)
            XrLrY = SP.dot(self.Xr.T,self.Y)
            self.cache['XrXrLrY'] = SP.dot(self.Xr,XrLrY) 
            self.cache['WrXrXrLrY'] = (self.cache['Sg']**0.5)[:,SP.newaxis]*SP.dot(self.cache['Vg'],XrLrY)

        if (self.Xr_has_changed or self.F_has_changed) and self.F is not None:
            """ F transf """
            self.cache['FF'] = SP.dot(self.F.T,self.F)
            self.cache['WrLrF'] = SP.dot(self.cache['Wr'],self.F)
            XrLrF = SP.dot(self.Xr.T,self.F)
            self.cache['XrXrLrF'] = SP.dot(self.Xr,XrLrF) 
            self.cache['FLrXrXrLrF'] = SP.dot(self.F.T,self.cache['XrXrLrF']) 
            self.cache['WrXrXrLrF'] = (self.cache['Sg']**0.5)[:,SP.newaxis]*SP.dot(self.cache['Vg'],XrLrF)

        if (self.F_has_changed or self.Y_has_changed) and self.F is not None:
            self.cache['FY'] = SP.dot(self.F.T,self.Y)

        if (self.Xr_has_changed or self.F_has_changed or self.Y_has_changed) and self.F is not None:
            self.cache['FXrXrLrY'] = SP.dot(self.F.T,self.cache['XrXrLrY'])

        if cov_params_have_changed or self.Y_has_changed:
            """ phenos transf """
            self.cache['LY'] = SP.dot(self.Y,self.cache['Lc'].T)
            self.cache['WrLY'] = SP.dot(self.cache['WrLrY'],self.cache['Lc'].T)
            self.cache['WLY'] = SP.dot(self.cache['WrLY'],self.cache['Wc'].T)
            self.cache['XrXrLY'] = SP.dot(self.cache['XrXrLrY'],self.cache['Lc'].T) 
            self.cache['WrXrXrLY'] = SP.dot(self.cache['WrXrXrLrY'],self.cache['Lc'].T) 

        if cov_params_have_changed and self.F is not None:
            """ A transf """
            # A for now is just I
            self.cache['LcA'] = self.cache['Lc']
            self.cache['Cni'] = SP.dot(self.cache['Lc'].T,self.cache['Lc'])
            self.cache['LcALcA'] = self.cache['Cni'] 
            self.cache['WcLcA'] = SP.dot(self.cache['Wc'],self.cache['LcA'])

        if cov_params_have_changed or self.Xr_has_changed or self.Y_has_changed:
            self.cache['DWLY'] = self.cache['D']*self.cache['WLY']
            self.cache['SgDWLY'] = self.cache['Sg'][:,SP.newaxis]*self.cache['DWLY']
            smartSum(self.time,'cache_colSVDpRot',TIME.time()-start)
            smartSum(self.count,'cache_colSVDpRot',1)

        if (cov_params_have_changed or self.Xr_has_changed or self.F_has_changed) and self.F is not None:
            self.cache['WLV'] = SP.kron(self.cache['WcLcA'],self.cache['WrLrF'])
            self.cache['DWLV'] = self.cache['d'][:,SP.newaxis]*self.cache['WLV']
            self.cache['DWLV_t'] = SP.reshape(self.cache['DWLV'],(self.S,self.rank,self.P*self.K),order='F')
            self.cache['SgDWLV_t'] = self.cache['Sg'][:,SP.newaxis,SP.newaxis]*self.cache['DWLV_t']
            self.cache['Areml'] = SP.kron(self.cache['LcALcA'],self.cache['FF'])
            self.cache['Areml']-= SP.dot(self.cache['WLV'].T,self.cache['DWLV'])
            self.cache['Areml_chol'] = LA.cholesky(self.cache['Areml']).T 
            # TODO: handle pseudo inverses
            self.cache['Areml_inv'] = LA.cho_solve((self.cache['Areml_chol'],True),SP.eye(self.K*self.P))


        if (cov_params_have_changed or self.Xr_has_changed or self.Y_has_changed or self.F_has_changed) and self.F is not None:
            VKiY = SP.dot(self.cache['FY'],self.cache['Cni'])
            #TODO: have not controlled factorization in the following line
            VKiY-= SP.dot(SP.dot(self.cache['WrLrF'].T,self.cache['DWLY']),self.cache['WcLcA'])
            self.cache['b'] = SP.dot(self.cache['Areml_inv'],SP.reshape(VKiY,(VKiY.size,1),order='F')) 
            self.cache['B'] = SP.reshape(self.cache['b'],(self.K,self.P), order='F')
            self.cache['BLc'] = SP.dot(self.cache['B'],self.cache['Lc'].T)
            self.cache['BLcWc'] = SP.dot(self.cache['BLc'],self.cache['Wc'].T)
            self.cache['Z'] = self.Y-SP.dot(self.F,self.cache['B']) 
            self.cache['FZ'] = self.cache['FY']-SP.dot(self.cache['FF'],self.cache['B']) 
            self.cache['LZ'] = self.cache['LY']-SP.dot(self.F,self.cache['BLc'])
            self.cache['WrLZ'] = self.cache['WrLY']-SP.dot(self.cache['WrLrF'],self.cache['BLc'])
            self.cache['WLZ'] = self.cache['WLY']-SP.dot(self.cache['WrLrF'],self.cache['BLcWc'])
            self.cache['DWLZ'] = self.cache['D']*self.cache['WLZ'] 
            self.cache['SgDWLZ'] = self.cache['Sg'][:,SP.newaxis]*self.cache['DWLZ'] 
            self.cache['XrXrLZ'] = self.cache['XrXrLY']-SP.dot(self.cache['XrXrLrF'],self.cache['BLc'])
            self.cache['WrXrXrLZ'] = self.cache['WrXrXrLY']-SP.dot(self.cache['WrXrXrLrF'],self.cache['BLc'])
            VKiZ = SP.dot(self.cache['FZ'],self.cache['Cni'])
            VKiZ-= SP.dot(self.cache['WrLrF'].T,SP.dot(self.cache['DWLZ'],self.cache['WcLcA']))
            self.cache['vecVKiZ'] = SP.reshape(VKiZ,(self.K*self.P,1),order='F')

        if self.F is None:
            """ Then Z=Y """
            self.cache['LZ'] = self.cache['LY']
            self.cache['WLZ'] = self.cache['WLY']
            self.cache['DWLZ'] = self.cache['DWLY']
            self.cache['XrXrLZ'] = self.cache['XrXrLY']
            self.cache['SgDWLZ'] = self.cache['SgDWLY']
            self.cache['WrXrXrLZ'] = self.cache['WrXrXrLY']
            self.cache['WrLZ'] = self.cache['WrLY']

        self.Y_has_changed = False
        self.F_has_changed = False
        self.Xr_has_changed = False
        self.Cr.params_have_changed = False
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
        lml += SP.sum(SP.log(self.cache['s']))
        lml += self.S*SP.sum(SP.log(self.cache['Se']))
        lml += self.rank*SP.sum(SP.log(self.cache['Sg']))
        lml += self.N*SP.sum(SP.log(self.cache['Sn']))

        #3. quatratic term
        lml += SP.sum(self.cache['LZ']*self.cache['LZ'])
        lml -= SP.sum(self.cache['WLZ']*self.cache['DWLZ'])

        #4. reml term
        if self.F is not None:
            lml += 2*SP.log(SP.diag(self.cache['Areml_chol'])).sum()

        lml *= 0.5

        smartSum(self.time,'lml',TIME.time()-start)
        smartSum(self.count,'lml',1)

        return lml

    def LMLdebug(self):
        """
        LML function for debug
        """
        assert self.N*self.P<5000, 'gp2kronSum:: N*P>=5000'

        y = SP.reshape(self.Y,(self.N*self.P), order='F') 
        V = SP.kron(SP.eye(self.P),self.F)

        XX = SP.dot(self.Xr,self.Xr.T)
        K  = SP.kron(self.Cr.K(),XX)
        K += SP.kron(self.Cn.K()+self.offset*SP.eye(self.P),SP.eye(self.N))

        # inverse of K
        cholK = LA.cholesky(K)
        Ki = LA.cho_solve((cholK,False),SP.eye(self.N*self.P))

        # Areml and inverse
        Areml = SP.dot(V.T,SP.dot(Ki,V))
        cholAreml = LA.cholesky(Areml)
        Areml_i = LA.cho_solve((cholAreml,False),SP.eye(self.K*self.P))

        # effect sizes and z
        b = SP.dot(Areml_i,SP.dot(V.T,SP.dot(Ki,y)))
        z = y-SP.dot(V,b)
        Kiz = SP.dot(Ki,z)

        # lml
        lml  = y.shape[0]*SP.log(2*SP.pi)
        lml += 2*SP.log(SP.diag(cholK)).sum()
        lml += 2*SP.log(SP.diag(cholAreml)).sum()
        lml += SP.dot(z,Kiz)
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
        covars = ['Cr','Cn']
        for covar in covars:        
            RV[covar] = self._LMLgrad_covar(covar)
        return RV

    def _LMLgrad_covar(self,covar,**kw_args):
        """
        calculates LMLgrad for covariance parameters
        """
        # precompute some stuff
        if covar=='Cr':
            trR = self.cache['trXrXr'] 
            RLZ = self.cache['XrXrLZ']
            SrDWLZ = self.cache['SgDWLZ']
            WrRLZ = self.cache['WrXrXrLZ']
            diagSr = self.cache['Sg']
            n_params = self.Cr.getNumberParams()
            if self.F is not None:
                SrDWLY = self.cache['SgDWLY']
                WrRLY = self.cache['WrXrXrLY']
                SrDWLV_t = self.cache['SgDWLV_t']
                WrRLF = self.cache['WrXrXrLrF']
                FRF = self.cache['FLrXrXrLrF']
                FRLrY = self.cache['FXrXrLrY']
        elif covar=='Cn':
            trR = self.N
            RLZ = self.cache['LZ']
            SrDWLZ = self.cache['DWLZ']
            WrRLZ = self.cache['WrLZ']
            diagSr = SP.ones(self.S)
            n_params = self.Cn.getNumberParams()
            if self.F is not None:
                SrDWLY = self.cache['DWLY']
                WrRLY = self.cache['WrLY']
                SrDWLV = self.cache['DWLV']
                WrRLF = self.cache['WrLrF']
                SrDWLV_t = self.cache['DWLV_t']
                FRF = self.cache['FF']
                FRLrY = self.cache['FY']

        # fill gradient vector
        RV = SP.zeros(n_params)
        for i in range(n_params):

            #0. calc LCL
            start = TIME.time()
            if covar=='Cr':     C = self.Cr.Kgrad_param(i)
            elif covar=='Cn':   C = self.Cn.Kgrad_param(i)
            LCL = SP.dot(self.cache['Lc'],SP.dot(C,self.cache['Lc'].T))
            LLCLL = SP.dot(self.cache['Lc'].T,SP.dot(LCL,self.cache['Lc']))
            LCLW = SP.dot(LCL,self.cache['Wc'].T)
            WLCLW = SP.dot(self.cache['Wc'],LCLW)

            CoRLZ = SP.dot(RLZ,LCL.T)
            CoSrDWLZ = SP.dot(SrDWLZ,WLCLW.T)
            WCoRLZ = SP.dot(WrRLZ,LCLW)

            if self.F is not None:
                WcCLcA = SP.dot(SP.dot(self.cache['Wc'],LCL),self.cache['LcA'])
                CoSrDWLY = SP.dot(SrDWLY,WLCLW.T)
                DCoSrDWLY = self.cache['D']*CoSrDWLY
                WCoRLY = SP.dot(WrRLY,LCLW)
                DWCoRLY = self.cache['D']*WCoRLY

                #0a. grad of Areml
                if 1:
                    Areml_grad = SP.dot(SP.kron(WcCLcA,WrRLF).T,self.cache['DWLV'])
                else:
                    Areml_grad = SP.tensordot(SP.tensordot(WrRLF,self.cache['DWLV_t'],axes=(0,0)),WcCLcA,axes=(1,0))
                    # and then resize...
                Areml_grad+= Areml_grad.T
                Areml_grad-= SP.kron(LLCLL,FRF) #TODO: think about LLCLL
                CoSrDWLV_t = SP.tensordot(SrDWLV_t,WLCLW,axes=(1,1))
                Areml_grad-= SP.tensordot(self.cache['DWLV_t'],CoSrDWLV_t,axes=([0,1],[0,2]))

                #0b. grad of beta
                B_grad1 = -SP.dot(FRLrY,LLCLL)
                B_grad1-= SP.dot(SP.dot(self.cache['WrLrF'].T,DCoSrDWLY),self.cache['WcLcA'])
                B_grad1+= SP.dot(SP.dot(WrRLF.T,self.cache['DWLY']),WcCLcA)
                B_grad1+= SP.dot(SP.dot(self.cache['WrLrF'].T,DWCoRLY),self.cache['WcLcA'])
                b_grad = SP.reshape(B_grad1,(self.K*self.P,1),order='F')
                b_grad-= SP.dot(Areml_grad,self.cache['b'])
                b_grad = SP.dot(self.cache['Areml_inv'],b_grad)

            #1. der of log det
            start = TIME.time()
            trC = LCL.diagonal().sum()
            RV[i] = trC*trR
            RV[i]-= SP.dot(self.cache['d'],SP.kron(WLCLW.diagonal(),diagSr))
            smartSum(self.time,'lmlgrad_trace',TIME.time()-start)
            smartSum(self.count,'lmlgrad_trace',1)

            #2. der of quad form
            start = TIME.time()
            RV[i]-= SP.sum(self.cache['LZ']*CoRLZ)
            RV[i]-= SP.sum(self.cache['DWLZ']*CoSrDWLZ)
            RV[i]+= 2*SP.sum(self.cache['DWLZ']*WCoRLZ)
            if self.F is not None:
                RV[i]-= 2*SP.dot(self.cache['vecVKiZ'].T,b_grad)
            smartSum(self.time,'lmlgrad_quadform',TIME.time()-start)
            smartSum(self.count,'lmlgrad_quadform',1)

            if self.F is not None:
                #3. reml term
                RV[i] += (self.cache['Areml_inv']*Areml_grad).sum()

            RV[i] *= 0.5

        return RV

    def LMLgrad_debug(self,**kw_args):
        """
        LML gradient debug
        """
        RV = {}
        covars = ['Cr','Cn']
        for covar in covars:
            RV[covar] = self._LMLgrad_covar_debug(covar)
        return RV

    def _LMLgrad_covar_debug(self,covar):

        assert self.N*self.P<5000, 'gp2kronSum:: N*P>=5000'

        y = SP.reshape(self.Y,(self.N*self.P), order='F') 
        V = SP.kron(SP.eye(self.P),self.F)

        # calc K
        XX = SP.dot(self.Xr,self.Xr.T)
        K  = SP.kron(self.Cr.K(),XX)
        K += SP.kron(self.Cn.K()+self.offset*SP.eye(self.P),SP.eye(self.N))

        # inverse of K
        cholK = LA.cholesky(K)
        Ki = LA.cho_solve((cholK,False),SP.eye(self.N*self.P))

        # Areml and inverse
        KiV = SP.dot(Ki,V)
        Areml = SP.dot(V.T,KiV)
        cholAreml = LA.cholesky(Areml)
        Areml_i = LA.cho_solve((cholAreml,False),SP.eye(self.K*self.P))

        # effect sizes and z
        b = SP.dot(Areml_i,SP.dot(V.T,SP.dot(Ki,y)))
        Vb = SP.dot(V,b)
        z = y-Vb
        Kiz = SP.dot(Ki,z)
        Kiy = SP.dot(Ki,y)
        zKiV = SP.dot(z.T,KiV)

        if covar=='Cr':     n_params = self.Cr.getNumberParams()
        elif covar=='Cn':   n_params = self.Cn.getNumberParams()

        RV = SP.zeros(n_params)

        for i in range(n_params):
            #0. calc grad_i
            if covar=='Cr':
                C   = self.Cr.Kgrad_param(i)
                Kgrad  = SP.kron(C,XX)
            elif covar=='Cn':
                C   = self.Cn.Kgrad_param(i)
                Kgrad  = SP.kron(C,SP.eye(self.N))

            #0a. Areml grad
            Areml_grad = -SP.dot(KiV.T,SP.dot(Kgrad,KiV))

            #0b. beta grad
            b_grad = -SP.dot(Areml_i,SP.dot(Areml_grad,b))
            b_grad-= SP.dot(Areml_i,SP.dot(KiV.T,SP.dot(Kgrad,Kiy)))

            #1. der of log det
            RV[i]  = 0.5*(Ki*Kgrad).sum()
            
            #2. der of quad form
            RV[i] -= 0.5*(Kiz*SP.dot(Kgrad,Kiz)).sum()
            RV[i] += SP.dot(zKiV,b_grad).sum()

            #3. der of reml term
            RV[i]+= 0.5*(Areml_i*Areml_grad).sum() 

        return RV

