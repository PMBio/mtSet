import sys
sys.path.append('./../../..')
from mtSet.pycore.utils.utils import smartSum
from mtSet.pycore.mean import mean

import pdb
import numpy as NP
import scipy as SP
import scipy.linalg as LA
import time as TIME
from gp_base import GP

class gp2kronSum(GP):
 
    def __init__(self,mean,Cg,Cn,XX=None,S_XX=None,U_XX=None,offset=1e-4):
        """
        Y:      Phenotype matrix
        Cg:     LIMIX trait-to-trait covariance for genetic contribution
        Cn:     LIMIX trait-to-trait covariance for noise
        XX:     Matrix for fixed sample-to-sample covariance function
        """
        #cache init
        self.cache = {}
        # pheno
        self.setMean(mean)
        # colCovariances
        self.setColCovars(Cg,Cn)
        # row covars
        self.set_XX(XX,S_XX,U_XX)
        #offset for trait covariance matrices
        self.setOffset(offset)
        self.params = None
        # time
        self.time = {}
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

    def setColCovars(self,Cg,Cn):
        """
        set column covariances
        """
        # col covars
        self.Cg = Cg
        self.Cn = Cn

    def setMean(self,mean):
        """
        set gp mean
        """
        self.N, self.P = mean.getDimensions()
        self.mean = mean

    def setY(self,Y):
        """
        set gp mean
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

    def getParams(self):
        """
        get hper parameters
        """
        params = {}
        params['Cg'] = self.Cg.getParams()
        params['Cn'] = self.Cn.getParams()
        if 'mean' in self.params.keys():
            params['mean'] = self.mean.getParams()
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
        keys =self.params.keys()
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
        cov_params_have_changed = self.Cg.params_have_changed or self.Cn.params_have_changed

        if self.XX_has_changed:
            start = TIME.time()
            """ Row SVD Bg + Noise """
            self.cache['Srstar'],Urstar  = LA.eigh(self.XX)
            self.cache['Lr']   = Urstar.T
            self.mean.setRowRotation(Lr=self.cache['Lr'])

            smartSum(self.time,'cache_XXchanged',TIME.time()-start)
            smartSum(self.count,'cache_XXchanged',1)
        
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

        self.XX_has_changed = False
        self.Cg.params_have_changed = False
        self.Cn.params_have_changed = False

    def LML(self,params=None,*kw_args):
        """
        calculate LML
        """
        if params!=None:
            self.setParams(params)

        self._update_cache()

        start = TIME.time()

        #1. const term
        lml  = self.N*self.P*SP.log(2*SP.pi)

        #2. logdet term
        lml += SP.sum(SP.log(self.cache['Sc2']))*self.N + SP.log(self.cache['s']).sum()

        #3. quatratic term
        lml += (self.cache['LY']*self.cache['DLY']).sum()

        lml *= 0.5

        smartSum(self.time,'lml',TIME.time()-start)
        smartSum(self.count,'lml',1)

        return lml


    def LMLdebug(self):
        """
        LML function for debug
        """
        assert self.N*self.P<2000, 'gp2kronSum:: N*P>=2000'

        y  = SP.reshape(self.Y,(self.N*self.P), order='F') 

        K  = SP.kron(self.Cg.K(),self.XX)
        K += SP.kron(self.Cn.K()+self.offset*SP.eye(self.P),SP.eye(self.N))

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
        covars = ['Cg','Cn']
        for covar in covars:        
            RV[covar] = self._LMLgrad_covar(covar)
        if 'mean' in self.params.keys():
            RV['mean'] = self._LMLgrad_mean()
        return RV

    def _LMLgrad_covar(self,covar,**kw_args):
        """
        calculates LMLgrad for covariance parameters
        """
        # precompute some stuff
        if covar=='Cg':
            LRLdiag = self.cache['Srstar']
            n_params = self.Cg.getNumberParams()
        elif covar=='Cn':
            LRLdiag = SP.ones(self.N)
            n_params = self.Cn.getNumberParams()

        # fill gradient vector
        RV = SP.zeros(n_params)
        for i in range(n_params):

            #0. calc LCL
            start = TIME.time()
            if covar=='Cr':     C = self.Cr.Kgrad_param(i)
            elif covar=='Cg':   C = self.Cg.Kgrad_param(i)
            elif covar=='Cn':   C = self.Cn.Kgrad_param(i)
            LCL = SP.dot(self.cache['Lc'],SP.dot(C,self.cache['Lc'].T))

            #1. der of log det
            start = TIME.time()
            kronDiag  = SP.kron(LCL.diagonal(),LRLdiag)
            RV[i] = SP.dot(self.cache['d'],kronDiag)
            smartSum(self.time,'lmlgrad_trace',TIME.time()-start)
            smartSum(self.count,'lmlgrad_trace',1)

            #2. der of quad form
            start = TIME.time()
            KDLY  = LRLdiag[:,SP.newaxis]*SP.dot(self.cache['DLY'],LCL.T)
            RV[i] -= (self.cache['DLY']*KDLY).sum()
            smartSum(self.time,'lmlgrad_quadform',TIME.time()-start)
            smartSum(self.count,'lmlgrad_quadform',1)

            RV[i] *= 0.5

        return RV

    def _LMLgrad_mean(self):
        """ LMLgradient with respect to the mean params """
        n_params = self.params['mean'].shape[0]
        RV = SP.zeros(n_params)
        for i in range(n_params):
            dF = self.mean.getGradient(i)
            RV[i] = (dF*self.cache['DLY']).sum()
        return RV

    def LMLgrad_debug(self,**kw_args):
        """
        LML gradient debug
        """
        RV = {}
        covars = ['Cg','Cn']
        for covar in covars:
            RV[covar] = self._LMLgrad_covar_debug(covar)
        return RV

    def _LMLgrad_covar_debug(self,covar):

        assert self.N*self.P<2000, 'gp2kronSum:: N*P>=2000'

        y  = SP.reshape(self.Y,(self.N*self.P), order='F') 

        K  = SP.kron(self.Cg.K(),self.XX)
        K += SP.kron(self.Cn.K()+self.offset*SP.eye(self.P),SP.eye(self.N))

        cholK = LA.cholesky(K).T
        Ki  = LA.cho_solve((cholK,True),SP.eye(y.shape[0]))
        Kiy   = LA.cho_solve((cholK,True),y)

        if covar=='Cr':     n_params = self.Cr.getNumberParams()
        elif covar=='Cg':   n_params = self.Cg.getNumberParams()
        elif covar=='Cn':   n_params = self.Cn.getNumberParams()

        RV = SP.zeros(n_params)

        for i in range(n_params):
            #0. calc grad_i
            if covar=='Cg':
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

    def predict(self,XXstar):
        """
        Make predictions:
            XXstar:     cross covariance matrix Nstar,N
        """
        self._update_cache()
        KiY = SP.dot(self.cache['Lr'].T,SP.dot(self.cache['DLY'],self.cache['Lc']))
        rv = SP.dot(XXstar,SP.dot(KiY,self.Cg.K()))
        return rv

    def ste(self,covar):
        """ calculates standard errors """
        varMLE, info = self.varMLE()
        RV = SP.zeros((self.P,self.P))
        idx1 = 0
        for key1 in ['Cg','Cn']:
            for key1_p1 in range(self.P):
                for key1_p2 in range(key1_p1,self.P):
                    idx2 = 0
                    for key2 in ['Cg','Cn']:
                        for key2_p1 in range(self.P):
                            for key2_p2 in range(key2_p1,self.P):
                                same_cov =  key1==covar and key2==covar
                                same_el = key1_p1==key2_p1 and key1_p2==key2_p2
                                if same_cov and same_el:
                                    RV[key1_p1,key1_p2] = RV[key1_p2,key1_p1] = SP.sqrt(varMLE[idx1,idx2])
                                idx2+=1
                    idx1+=1
        return RV

    """ getting heritabilities and standard errors """

    def h2(self):
        """ heritability """
        self._update_cache()
        vg = self.Cg.K().diagonal()
        vn = self.Cn.K().diagonal()
        rv = vg/(vg+vn)
        return rv

    def h2_ste(self):
        """
        standard errors over heritability
        h2_ste = sqrt((dh2_dvg)^2 vg + (dh2_dvn)^2 vn +
                        (dh2_dvg)(dh2_dvn) v_gn)
        """
        vg = self.Cg.K().diagonal() 
        vn = self.Cn.K().diagonal() 
        s2_vg = self.ste('Cg').diagonal()**2
        s2_vn = self.ste('Cn').diagonal()**2
        s_vgvn = self.sigma_vg_vn()
        rv = vn**2/(vg+vn)**4*s2_vg
        rv+= vg**2/(vg+vn)**4*s2_vn
        rv+= 2*vg*vn/(vg+vn)**4*s_vgvn
        rv = SP.sqrt(rv)
        return rv

    def sigma_vg_vn(self):
        """ calculates standard errors """
        varMLE, info = self.varMLE()
        RV = SP.zeros(self.P)
        idx1 = 0
        for key1 in ['Cg','Cn']:
            for key1_p1 in range(self.P):
                for key1_p2 in range(key1_p1,self.P):
                    idx2 = 0
                    for key2 in ['Cg','Cn']:
                        for key2_p1 in range(self.P):
                            for key2_p2 in range(key2_p1,self.P):
                                cov_cond =  key1=='Cg' and key2=='Cn'
                                el_cond = key1_p1==key1_p2 and key2_p1==key2_p2 and key1_p1==key2_p1
                                if cov_cond and el_cond:
                                    RV[key1_p1] = varMLE[idx1,idx2]
                                idx2+=1
                    idx1+=1
        return RV

    def varMLE(self):
        """ calculate inverse of fisher information """
        self._update_cache()
        Sr = {}
        Sr['Cg'] = self.cache['Srstar']
        Sr['Cn'] = SP.ones(self.N)
        n_params = self.Cg.getNumberParams()+self.Cn.getNumberParams()
        fisher = SP.zeros((n_params,n_params))
        header = SP.zeros((n_params,n_params),dtype='|S10')
        C1  = SP.zeros((self.P,self.P))
        C2  = SP.zeros((self.P,self.P))
        idx1 = 0
        for key1 in ['Cg','Cn']:
            for key1_p1 in range(self.P):
                for key1_p2 in range(key1_p1,self.P):
                    C1[key1_p1,key1_p2] = C1[key1_p2,key1_p1] = 1
                    LCL1 = SP.dot(self.cache['Lc'],SP.dot(C1,self.cache['Lc'].T))
                    CSr1 = SP.kron(Sr[key1][:,SP.newaxis,SP.newaxis],LCL1[SP.newaxis,:])
                    DCSr1 = self.cache['D'][:,:,SP.newaxis]*CSr1
                    idx2 = 0
                    for key2 in ['Cg','Cn']:
                        for key2_p1 in range(self.P):
                            for key2_p2 in range(key2_p1,self.P):
                                C2[key2_p1,key2_p2] = C2[key2_p2,key2_p1] = 1
                                LCL2 = SP.dot(self.cache['Lc'],SP.dot(C2,self.cache['Lc'].T))
                                CSr2 = SP.kron(Sr[key2][:,SP.newaxis,SP.newaxis],LCL2[SP.newaxis,:])
                                DCSr2 = self.cache['D'][:,:,SP.newaxis]*CSr2
                                fisher[idx1,idx2] = 0.5*(DCSr1*DCSr2).sum()
                                header[idx1,idx2] = '%s%d%d_%s%d%d'%(key1,key1_p1,key1_p1,key2,key2_p1,key2_p2)
                                C2[key2_p1,key2_p2] = C2[key2_p2,key2_p1] = 0
                                idx2+=1
                    C1[key1_p1,key1_p2] = C1[key1_p2,key1_p1] = 0
                    idx1+=1
        RV = LA.inv(fisher)
        return RV,header

    def ste_old(self,covar):
        """ calculates standard errors """
        self._update_cache()
        if covar=='Cg':
            Sr = self.cache['Srstar']
        elif covar=='Cn':
            Sr = SP.ones(self.N)
        RV = SP.zeros((self.P,self.P))
        C  = SP.zeros((self.P,self.P))
        for p1 in range(self.P):
            for p2 in range(p1,self.P):
                C[p1,p2] = C[p2,p1] = 1
                LCL = SP.dot(self.cache['Lc'],SP.dot(C,self.cache['Lc'].T))
                #fisher information
                CSr = SP.kron(Sr[:,SP.newaxis,SP.newaxis],LCL[SP.newaxis,:])
                DCSr = self.cache['D'][:,:,SP.newaxis]*CSr
                RV[p1,p2] = RV[p2,p1] = 1/SP.sqrt(0.5*(DCSr**2).sum())
                C[p1,p2] = C[p2,p1] = 0
        return RV

    def ste_old_debug(self,covar):
        """ debug function for standard errors """
        self._update_cache()
        if covar=='Cg':
            Sr = self.cache['Srstar']
        elif covar=='Cn':
            Sr = SP.ones(self.N)
        RV = SP.zeros((self.P,self.P))
        C  = SP.zeros((self.P,self.P))
        for p1 in range(self.P):
            for p2 in range(p1,self.P):
                C[p1,p2] = C[p2,p1] = 1
                LCL = SP.dot(self.cache['Lc'],SP.dot(C,self.cache['Lc'].T))
                #fisher information
                DCSr = SP.dot(SP.diag(self.cache['d']),SP.kron(LCL,SP.diag(Sr)))
                RV[p1,p2] = RV[p2,p1] = 1/SP.sqrt(0.5*(DCSr**2).sum())
                C[p1,p2] = C[p2,p1] = 0
        return RV

