import sys
sys.path.append('./../../..')
from mtSet.pycore.utils.utils import dumpDictHdf5
from mtSet.pycore.utils.utils import smartDumpDictHdf5
from mtSet.pycore.utils.fit_utils import fitPairwiseModel

# core
from mtSet.pycore.gp import gp3kronSum
from mtSet.pycore.gp import gp2kronSumLR
from mtSet.pycore.gp import gp2kronSum
from mtSet.pycore.mean import mean
import mtSet.pycore.covariance as covariance
import mtSet.pycore.optimize.optimize_bfgs as OPT

import h5py
import pdb
import scipy as SP
import scipy.stats as ST
import scipy.linalg as LA
import time as TIME
import copy
import warnings
import os


class MultiTraitSetTest():

    def __init__(self,Y=None,XX=None,S_XX=None,U_XX=None,traitID=None,colCovarType=None,rank_r=1,rank_g=1,rank_n=1,F=None):
        """
        Constructor
        Args:
            Y:              phenotypes
            XX:             genetic kinship, if is none no polygenic effect is considered
            colCovarType_r: column covariance matrix for genetic term
            colCovarType_n: column covariance matrix for noise term
            rank_r:         rank of region trait covar
            rank_g:         rank of genetic trait covar
            rank_n:         rank of noise trait covar
        """
        # assert
        assert Y is not None, 'MultiTraitSetTest:: set Y'

        #init fixed mean term
        self._initMean(Y,F=F)
        # data
        self.set_XX(XX,S_XX,U_XX)
        #traitID
        if traitID is None: traitID = SP.array(['trait%d'%(p+1) for p in range(self.P)])
        self.setTraitID(traitID)
        #init covariance matrices and gp
        self._initGP(colCovarType,rank_r,rank_g,rank_n)
        # null model params
        self.null = None
        # calls itself for column-by-column trait analysis
        self.mtssST = None
        self.nullST = None
        self.infoOpt   = None
        self.infoOptST = None
        pass

    def set_XX(self,XX=None,S_XX=None,U_XX=None):
        """
        set XX
        Args:
            XX:     fixed row covariance matrix
            S_XX:   eigenvalues of XX
            U_XX:   eigenvectors of XX
        """
        bothNone = S_XX is None and U_XX is None
        noneNone = S_XX is not None and U_XX is not None
        assert bothNone or noneNone, 'Please either specify both S_XX and U_XX or none of them'
        self.XX  = XX
        self.S_XX = S_XX
        self.U_XX = U_XX
        self.bgRE = self.XX is not None or noneNone

    def _initMean(self,Y,F=None,tol=1e-6):
        """
        initialize the mean term
        Args:
            F:    sample design of the fixed effect
        """
        if F is not None:
            R = LA.qr(F,mode='r')[0][:F.shape[1],:]
            I = (abs(R.diagonal())>tol)
            if SP.any(~I):
                warnings.warn('cols '+str(SP.where(~I)[0])+' have been removed because linearly dependent on the others')
            self.F = F[:,I]
        else:
            self.F = None
        #dimensions
        self.N,self.P = Y.shape
        #get F and Y
        self.Y=Y
        # build mean
        self.mean = mean(Y)
        if F is not None:
            A = SP.eye(self.P)
            self.mean.addFixedEffect(F=self.F,A=A)

    def _initGP(self,colCovarType,rank_r,rank_g,rank_n):
        """
        Initializes genetic and noise LIMIX trait covar
        Args:
            colCovarType_g:     genetic trait covariance type
            colCovarType_n:     noise trait covariance type
            rank_r:     rank of region trait covar
            rank_g:     rank of genetic trait covar
            rank_n:     rank of noise trait covar
        """
        if self.P==1:
            self.rank_r = 1
            self.rank_g = 1
            self.rank_n = 1
            # ideally should be diag
            colCovarType = 'freeform'
        elif colCovarType is None:
            colCovarType='freeform'

        self.rank_r = rank_r
        self.rank_g = rank_g
        self.rank_n = rank_n
        self.colCovarType = colCovarType
        self.Cg = self._buildTraitCovar(colCovarType,rank_g)
        self.Cn = self._buildTraitCovar(colCovarType,rank_n)
        XXnotNone = self.XX is not None
        SUnotNone = self.S_XX is not None and self.U_XX is not None
        # build mean
        if self.bgRE:
            self.gp = gp3kronSum(self.mean,self.Cg,self.Cn,XX=self.XX,S_XX=self.S_XX,U_XX=self.U_XX,rank=self.rank_r)
        else:
            self.gp = gp2kronSumLR(self.Y,self.Cn,F=self.F,rank=self.rank_r)

    def setTraitID(self,traitID):
        """ set trait id """
        assert traitID.shape[0]==self.P, 'MultiTraitSetTest:: dimension dismatch'
        self.traitID = traitID

    def _setY(self,Y):
        """ internal function: set pheno """
        assert Y.shape[0]==self.N, 'MultiTraitSetTest:: dimension dismatch'
        assert Y.shape[1]==self.P, 'MultiTraitSetTest:: dimension dismatch'
        self.Y = Y
        self.gp.setY(self.Y)

    def fitNull(self,verbose=True,cache=False,out_dir='./cache',fname=None,rewrite=False,seed=None,n_times=10,factr=1e3,init_method=None):
        """
        Fit null model
        """
        if seed is not None:    SP.random.seed(seed)

        read_from_file = False
        if cache:
            assert fname is not None, 'MultiTraitSetTest:: specify fname'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_file = os.path.join(out_dir,fname)
            read_from_file = os.path.exists(out_file) and not rewrite

        RV = {}
        if read_from_file:
            f = h5py.File(out_file,'r')
            for key in f.keys():
                RV[key] = f[key][:]
            f.close()
            self.setNull(RV)
        else:
            start = TIME.time()
            if self.bgRE:
                self.gpNull = gp2kronSum(self.mean,self.Cg,self.Cn,XX=self.XX,S_XX=self.S_XX,U_XX=self.U_XX)
            else:
                self.gpNull = gp2kronSumLR(self.Y,self.Cn,Xr=SP.ones((self.N,1)),F=self.F)
            for i in range(n_times):
                params0,Ifilter=self._initParams(init_method=init_method)
                conv,info = OPT.opt_hyper(self.gpNull,params0,Ifilter=Ifilter,factr=factr)
                if conv: break
            if not conv:    warnings.warn("not converged")
            LMLgrad = SP.concatenate([self.gpNull.LMLgrad()[key]**2 for key in self.gpNull.LMLgrad().keys()]).mean()
            LML = self.gpNull.LML()
            if 'mean' in params0.keys():
                RV['params_mean'] = self.gpNull.mean.getParams()
            RV['params0_g'] = self.Cg.getParams()
            RV['params0_n'] = self.Cn.getParams()
            RV['Cg'] = self.Cg.K()
            RV['Cn'] = self.Cn.K()
            RV['conv'] = SP.array([conv])
            RV['time'] = SP.array([TIME.time()-start])
            RV['NLL0'] = SP.array([LML])
            RV['LMLgrad'] = SP.array([LMLgrad])
            RV['nit'] = SP.array([info['nit']])
            RV['funcalls'] = SP.array([info['funcalls']])
            if self.bgRE:
                RV['h2'] = self.gpNull.h2()
                RV['h2_ste'] = self.gpNull.h2_ste()
                RV['Cg_ste'] = self.gpNull.ste('Cg')
                RV['Cn_ste'] = self.gpNull.ste('Cn')
            self.null = RV
            if cache:
                f = h5py.File(out_file,'w')
                dumpDictHdf5(RV,f)
                f.close()
        return RV

    def getNull(self):
        """ get null model info """
        return self.null

    def setNull(self,null):
        """ set null model info """
        self.null = null

    def optimize(self,Xr,params0=None,n_times=10,verbose=True,vmax=5,perturb=1e-3,factr=1e7):
        """
        Optimize the model considering Xr
        """
        # set params0 from null if params0==Null
        if params0 is None:
            if self.null is None:
                if verbose:     print ".. fitting null model upstream"
                self.fitNull()
            if self.bgRE:
                params0 = {'Cg':self.null['params0_g'],'Cn':self.null['params0_n']}
            else:
                params0 = {'Cn':self.null['params0_n']}
            if 'params_mean' in self.null:
                if self.null['params_mean'].shape[0]>0:
                    params0['mean'] = self.null['params_mean']
            params_was_None = True
        else:
            params_was_None = False
        Xr *= SP.sqrt(self.N/(Xr**2).sum())
        self.gp.set_Xr(Xr)
        self.gp.restart()
        start = TIME.time()
        for i in range(n_times):
            if params_was_None:
                params0['Cr'] = 1e-3*SP.randn(self.rank_r*self.P)
            conv,info = OPT.opt_hyper(self.gp,params0,factr=factr)
            conv *= self.gp.Cr.K().diagonal().max()<vmax
            conv *= self.getLMLgrad()<0.1
            if conv or not params_was_None: break
        self.infoOpt = info
        if not conv:
            warnings.warn("not converged")
        # return value
        RV = {}
        if self.P>1:
            RV['Cr']  = self.getCr()
            if self.bgRE: RV['Cg']  = self.getCg()
            RV['Cn']  = self.getCn()
        RV['time']  = SP.array([TIME.time()-start])
        RV['params0'] = params0
        RV['nit'] = SP.array([info['nit']])
        RV['funcalls'] = SP.array([info['funcalls']])
        RV['var']    = self.getVariances()
        RV['conv']  = SP.array([conv])
        RV['NLLAlt']  = SP.array([self.getNLLAlt()])
        RV['LLR']    = SP.array([self.getLLR()])
        RV['LMLgrad'] = SP.array([self.getLMLgrad()])
        return RV

    def getInfoOpt(self):
        """ get information for the optimization """
        return self.infoOpt

    def getTimeProfiling(self):
        """ get time profiling """
        rv = {'time':self.gp.get_time(),'count':self.gp.get_count()}
        return rv

    def getCr(self):
        """
        get estimated region trait covariance
        """
        assert self.P>1, 'this is a multitrait model'
        return self.gp.Cr.K()
        
    def getCg(self):
        """
        get estimated genetic trait covariance
        """
        assert self.P>1, 'this is a multitrait model'
        return self.gp.Cg.K()

    def getCn(self):
        """
        get estimated noise trait covariance
        """
        assert self.P>1, 'this is a multitrait model'
        return self.gp.Cn.K()

    def getVariances(self):
        """
        get variances
        """
        if self.P==1:
            params = self.gp.getParams()
            if self.bgRE:       keys = ['Cr','Cg','Cn']
            else:               keys = ['Cr','Cn']
            var = SP.array([params[key][0]**2 for key in keys])
        else:
            var = []
            var.append(self.getCr().diagonal())
            if self.bgRE:
                var.append(self.getCg().diagonal())
            var.append(self.getCn().diagonal())
            var = SP.array(var)
        return var

    def getNLLAlt(self):
        """
        get negative log likelihood of the alternative
        """
        return self.gp.LML()

    def getLLR(self):
        """
        get log likelihood ratio
        """
        assert self.null is not None, 'null model needs to be fitted!'
        return self.null['NLL0'][0]-self.getNLLAlt()

    def getLMLgrad(self):
        """
        get norm LML gradient
        """
        LMLgrad = self.gp.LMLgrad()
        lmlgrad  = 0
        n_params = 0
        for key in LMLgrad.keys():
            lmlgrad  += (LMLgrad[key]**2).sum()
            n_params += LMLgrad[key].shape[0]
        lmlgrad /= float(n_params)
        return lmlgrad

    def fitNullTraitByTrait(self,verbose=True,cache=False,out_dir='./cache',fname=None,rewrite=False):
        """
        Fit null model trait by trait
        """
        read_from_file = False
        if cache:
            assert fname is not None, 'MultiTraitSetTest:: specify fname'
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            out_file = os.path.join(out_dir,fname)
            read_from_file = os.path.exists(out_file) and not rewrite

        RV = {}
        if read_from_file:
            f = h5py.File(out_file,'r')
            for p in range(self.P):
                trait_id = self.traitID[p]
                g = f[trait_id]
                RV[trait_id] = {}
                for key in g.keys():
                    RV[trait_id][key] = g[key][:]
            f.close()
            self.nullST=RV
        else:
            """ create mtssST and fit null column by column returns all info """
            if self.mtssST is None:
                y = SP.zeros((self.N,1)) 
                self.mtssST = MultiTraitSetTest(y,XX=self.XX,S_XX=self.S_XX,U_XX=self.U_XX,F=self.F)
            RV = {}
            for p in range(self.P):
                trait_id = self.traitID[p]
                y = self.Y[:,p:p+1]
                self.mtssST._setY(y)
                RV[trait_id] = self.mtssST.fitNull()
            self.nullST = RV
            if cache:
                f = h5py.File(out_file,'w')
                smartDumpDictHdf5(RV,f)
                f.close()
        return RV

    def optimizeTraitByTrait(self,Xr,verbose=True,n_times=10,factr=1e7):
        """ Optimize trait by trait """
        assert self.nullST is not None, 'fit null model beforehand'
        if self.mtssST is None:
            y = SP.zeros((self.N,1)) 
            self.mtssST = MultiTraitSetTest(y,XX=self.XX,S_XX=self.S_XX,U_XX=self.U_XX,F=self.F)
        RV = {}
        self.infoOptST = {}
        self.timeProfilingST = {}
        for p in range(self.P):
            y = self.Y[:,p:p+1]
            trait_id = self.traitID[p]
            self.mtssST._setY(y)
            self.mtssST.setNull(self.nullST[trait_id])
            RV[trait_id] = self.mtssST.optimize(Xr,n_times=n_times,factr=factr)
            self.infoOptST[trait_id] = self.mtssST.getInfoOpt()
            self.timeProfilingST[trait_id] = self.mtssST.getTimeProfiling()
        return RV

    def getInfoOptST(self):
        """ get information for the optimization """
        return self.infoOptST

    def getTimeProfilingST(self):
        """ get time profiling """
        return self.timeProfilingST

    def _buildTraitCovar(self,trait_covar_type,rank=1):
        """
        Internal functions that builds the trait covariance matrix using the LIMIX framework

        Args:
            trait_covar_type:   type of covaraince to use in {freeform,lowrank_id,lowrank_diag}
            rank:               rank of a possible lowrank component (default 1)
        Returns:
            LIMIX::PCovarianceFunction for Trait covariance matrix
        """
        assert trait_covar_type in 'freeform', '%s not supported yet'%trait_covar_type
        cov = covariance.freeform(self.P)
        return cov

    def _initParams(self,init_method=None):
        """ this function initializes the paramenter and Ifilter """
        if self.P==1:
            if self.bgRE:
                params0 = {'Cg':SP.sqrt(0.5)*SP.ones(1),'Cn':SP.sqrt(0.5)*SP.ones(1)}
                Ifilter = None
            else:
                params0 = {'Cr':1e-9*SP.ones(1),'Cn':SP.ones(1)}
                Ifilter = {'Cr':SP.zeros(1,dtype=bool),'Cn':SP.ones(1,dtype=bool)}
        else:
            if self.bgRE:
                if self.colCovarType=='freeform':
                    if init_method=='pairwise':
                        _RV = fitPairwiseModel(self.Y,XX=self.XX,S_XX=self.S_XX,U_XX=self.U_XX,verbose=False)
                        params0 = {'Cg':_RV['params0_Cg'],'Cn':_RV['params0_Cn']}
                    elif init_method=='random':
                        params0 = {'Cg':SP.randn(self.Cg.getNumberParams()),'Cn':SP.randn(self.Cn.getNumberParams())}
                    else:
                        cov = 0.5*SP.cov(self.Y.T)+1e-4*SP.eye(self.P)
                        chol = LA.cholesky(cov,lower=True)
                        params = chol[SP.tril_indices(self.P)]
                        params0 = {'Cg':params.copy(),'Cn':params.copy()}
                Ifilter = None
            else:
                if self.colCovarType=='freeform':
                    cov = SP.cov(self.Y.T)+1e-4*SP.eye(self.P)
                    chol = LA.cholesky(cov,lower=True)
                    params = chol[SP.tril_indices(self.P)]
                #else:
                #    S,U=LA.eigh(cov)
                #    a = SP.sqrt(S[-self.rank_r:])[:,SP.newaxis]*U[:,-self.rank_r:]
                #    if self.colCovarType=='lowrank_id':
                #        c = SP.sqrt(S[:-self.rank_r].mean())*SP.ones(1)
                #    else:
                #        c = SP.sqrt(S[:-self.rank_r].mean())*SP.ones(self.P)
                #    params0_Cn = SP.concatenate([a.T.ravel(),c])
                params0 = {'Cr':1e-9*SP.ones(self.P),'Cn':params}
                Ifilter = {'Cr':SP.zeros(self.P,dtype=bool),
                            'Cn':SP.ones(params.shape[0],dtype=bool)}
        if self.mean.F is not None and self.bgRE:
            params0['mean'] = 1e-6*SP.randn(self.mean.getParams().shape[0])
            if Ifilter is not None:
                Ifilter['mean'] = SP.ones(self.mean.getParams().shape[0],dtype=bool)
        return params0,Ifilter

