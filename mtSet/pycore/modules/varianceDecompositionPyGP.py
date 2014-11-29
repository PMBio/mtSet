import sys
sys.path.append('./../../..')
import copy

from mtSet.pycore.mean import mean
import mtSet.pycore.gp.gp2kronSum as gp2kronSum
import mtSet.pycore.optimize.optimize_bfgs as OPT
import limix
import scipy as SP
import scipy.linalg as LA
import h5py
import pdb
import pylab as PL
import copy
import os
import cPickle

class VarianceDecompositionPyGP():
    """ This class """

    def __init__(self,Y,XX):
        """ Constructor """
        self.Y  = Y
        self.XX = XX
        self.N,self.P = Y.shape
        self.cache={}
        self.cache['singleTrait'] = None
        self.cache['pairwise']    = None
        self.cache['diagonal']    = None
        self.cache['freeform']    = None

    def fitSingleTraitModel(self,verbose=False):
        """ fit single trait model """
        if self.cache['singleTrait']==None:
            RV = {}
            Cg = limix.CDiagonalCF(1)
            Cn = limix.CDiagonalCF(1)
            gp = gp2kronSum(mean(self.Y[:,0:1]),Cg,Cn,self.XX)
            params0 = {'Cg':SP.sqrt(0.5)*SP.ones(1),'Cn':SP.sqrt(0.5)*SP.ones(1)}
            var = SP.zeros((self.P,2))
            conv1 = SP.zeros(self.P,dtype=bool)
            for p in range(self.P):
                if verbose:
                    print '.. fitting variance trait %d'%p
                gp.setY(self.Y[:,p:p+1])
                conv1[p],info = OPT.opt_hyper(gp,params0,factr=1e3)
                var[p,0] = Cg.K()[0,0]
                var[p,1] = Cn.K()[0,0]
            RV['conv1'] = conv1
            RV['varST'] = var
            self.cache['singleTrait'] = copy.deepcopy(RV)
        else:
            RV = copy.deepcopy(self.cache['singleTrait'])
        return RV

    def fitPairwiseModel(self,verbose=False):
        """ initilizes parameters """
        if self.cache['pairwise']==None:
            RV = self.fitSingleTraitModel(verbose=verbose)
            Cg = limix.CFreeFormCF(2)
            Cn = limix.CFreeFormCF(2)
            gp = gp2kronSum(mean(self.Y[:,0:2]),Cg,Cn,self.XX)
            conv2 = SP.ones((self.P,self.P),dtype=bool)
            rho_g = SP.ones((self.P,self.P))
            rho_n = SP.ones((self.P,self.P))
            for p1 in range(self.P):
                for p2 in range(p1):
                    if verbose:
                        print '.. fitting correlation (%d,%d)'%(p1,p2)
                    gp.setY(self.Y[:,[p1,p2]])
                    Cg_params0 = SP.array([SP.sqrt(RV['varST'][p1,0]),1e-6*SP.randn(),SP.sqrt(RV['varST'][p2,0])])
                    Cn_params0 = SP.array([SP.sqrt(RV['varST'][p1,1]),1e-6*SP.randn(),SP.sqrt(RV['varST'][p2,1])])
                    params0 = {'Cg':Cg_params0,'Cn':Cn_params0}
                    conv2[p1,p2],info = OPT.opt_hyper(gp,params0,factr=1e3)
                    rho_g[p1,p2] = Cg.K()[0,1]/SP.sqrt(Cg.K().diagonal().prod())
                    rho_n[p1,p2] = Cn.K()[0,1]/SP.sqrt(Cn.K().diagonal().prod())
                    conv2[p2,p1] = conv2[p1,p2]; rho_g[p2,p1] = rho_g[p1,p2]; rho_n[p2,p1] = rho_n[p1,p2]
            RV['Cg0'] = rho_g*SP.dot(SP.sqrt(RV['varST'][:,0:1]),SP.sqrt(RV['varST'][:,0:1].T))
            RV['Cn0'] = rho_n*SP.dot(SP.sqrt(RV['varST'][:,1:2]),SP.sqrt(RV['varST'][:,1:2].T))
            RV['conv2'] = conv2
            #3. regularizes covariance matrices
            offset_g = abs(SP.minimum(LA.eigh(RV['Cg0'])[0].min(),0))+1e-4
            offset_n = abs(SP.minimum(LA.eigh(RV['Cn0'])[0].min(),0))+1e-4
            RV['Cg0_reg'] = RV['Cg0']+offset_g*SP.eye(self.P)
            RV['Cn0_reg'] = RV['Cn0']+offset_n*SP.eye(self.P)
            self.cache['pairwise'] = copy.deepcopy(RV)
        else:
            RV = copy.deepcopy(self.cache['pairwise'])

        return RV

    def fit(self,model='freeform',lambd=0.5,rank=1,XXstar=None,verbose=False):
        if model=='freeform':
            RV = self.fit_freeform(XXstar=XXstar,verbose=verbose)
        elif model=='lowrank_diag':
            RV = self.fit_lowrank_diag(rank=rank,XXstar=XXstar,verbose=verbose)
        elif model=='lrd_geno':
            RV = self.fit_lrd_geno(rank=rank,XXstar=XXstar,verbose=verbose)
        elif model=='diagonal':
            RV = self.fit_diagonal(XXstar=XXstar,verbose=verbose)
        elif model=='freeformPen':
            RV = self.fit_freeformPen(lambd=lambd,XXstar=XXstar,verbose=verbose)
        return RV

    def fit_freeform(self,XXstar=None,verbose=False):
        """ fit a freeform model """
        if self.cache['freeform']==None:
            #1. initialize params
            RV = self.fitPairwiseModel(verbose=verbose)

            #2. init gp and params
            Cg = limix.CFreeFormCF(self.P)
            Cn = limix.CFreeFormCF(self.P)
            offset_g = abs(SP.minimum(LA.eigh(RV['Cg0'])[0].min(),0))+1e-4
            offset_n = abs(SP.minimum(LA.eigh(RV['Cn0'])[0].min(),0))+1e-4
            Lg = LA.cholesky(RV['Cg0']+offset_g*SP.eye(self.P))
            Ln = LA.cholesky(RV['Cn0']+offset_n*SP.eye(self.P))
            Cg_params0 = SP.concatenate([Lg[:,p][:p+1] for p in range(self.P)])
            Cn_params0 = SP.concatenate([Ln[:,p][:p+1] for p in range(self.P)])
            self.gp = gp2kronSum(self.Y,Cg,Cn,self.XX)
            params0 = {'Cg':Cg_params0,'Cn':Cn_params0}

            #3.  fit model
            if verbose:
                print '.. fitting freeform model'
            conv,info = OPT.opt_hyper(self.gp,params0,factr=1e3)

            #4. output
            if XXstar!=None:
                RV['Ystar'] = self.gp.predict(XXstar)
            RV['Cg'] = Cg.K()
            RV['Cn'] = Cn.K()
            RV['conv'] = SP.array([conv])
            RV['LMLgrad'] = SP.array([(info['grad']**2).sum()])
            self.cache['freeform'] = copy.deepcopy(RV)
        else:
            RV = copy.deepcopy(self.cache['freeform'])

        return RV

    def fit_lowrank_diag(self,rank=1,XXstar=None,verbose=False):
        """ fit a freeform model """
        #1. initialize params
        RV = self.fitPairwiseModel(verbose=verbose)

        #2. init params
        Sg,Ug = LA.eigh(RV['Cg0'])
        Sn,Un = LA.eigh(RV['Cn0'])
        Ag = SP.sqrt(Sg[-rank:])*Ug[:,-rank:]
        An = SP.sqrt(Sn[-rank:])*Un[:,-rank:]
        cg  = SP.sqrt(SP.maximum(SP.sqrt(Sg[:-rank].mean()),1e-4))*SP.ones(self.P)
        cn  = SP.sqrt(SP.maximum(SP.sqrt(Sn[:-rank].mean()),1e-4))*SP.ones(self.P)
        Cg_params0 = SP.concatenate([Ag.reshape(self.P*rank,order='F'),cg])
        Cn_params0 = SP.concatenate([An.reshape(self.P*rank,order='F'),cn])

        # covariance matrix
        Cg = limix.CSumCF()
        Cg.addCovariance(limix.CLowRankCF(self.P,rank))
        Cg.addCovariance(limix.CDiagonalCF(self.P))
        Cn = limix.CSumCF()
        Cn.addCovariance(limix.CLowRankCF(self.P,rank))
        Cn.addCovariance(limix.CDiagonalCF(self.P))
        
        # init gp
        self.gp = gp2kronSum(self.Y,Cg,Cn,self.XX)
        params0 = {'Cg':Cg_params0,'Cn':Cn_params0}

        #3.  fit model
        if verbose:
            print '.. fitting lowrank model with rank %d'%rank
        conv,info = OPT.opt_hyper(self.gp,params0,factr=1e3)

        #4. output
        if XXstar!=None:
            RV['Ystar'] = self.gp.predict(XXstar)
        RV['Cg'] = Cg.K()
        RV['Cn'] = Cn.K()
        RV['conv'] = SP.array([conv])
        RV['LMLgrad'] = SP.array([(info['grad']**2).sum()])

        return RV

    def fit_lrd_geno(self,rank=1,XXstar=None,verbose=False):
        """ fit a freeform model """
        #1. initialize params
        RV = self.fitPairwiseModel(verbose=verbose)

        #2. init params
        Sg,Ug = LA.eigh(RV['Cg0'])
        Ag = SP.sqrt(Sg[-rank:])*Ug[:,-rank:]
        cg  = SP.sqrt(SP.maximum(SP.sqrt(Sg[:-rank].mean()),1e-4))*SP.ones(self.P)
        offset_n = abs(SP.minimum(LA.eigh(RV['Cn0'])[0].min(),0))+1e-4
        Ln = LA.cholesky(RV['Cn0']+offset_n*SP.eye(self.P))
        Cg_params0 = SP.concatenate([Ag.reshape(self.P*rank,order='F'),cg])
        Cn_params0 = SP.concatenate([Ln[:,p][:p+1] for p in range(self.P)])

        # covariance matrix
        Cg = limix.CSumCF()
        Cg.addCovariance(limix.CLowRankCF(self.P,rank))
        Cg.addCovariance(limix.CDiagonalCF(self.P))
        Cn = limix.CFreeFormCF(self.P)
        
        # init gp
        self.gp = gp2kronSum(self.Y,Cg,Cn,self.XX)
        params0 = {'Cg':Cg_params0,'Cn':Cn_params0}

        #3.  fit model
        if verbose:
            print '.. fitting lowrank model only for geno with rank %d'%rank
        conv,info = OPT.opt_hyper(self.gp,params0,factr=1e3)

        #4. output
        if XXstar!=None:
            RV['Ystar'] = self.gp.predict(XXstar)
        RV['Cg'] = Cg.K()
        RV['Cn'] = Cn.K()
        RV['conv'] = SP.array([conv])
        RV['LMLgrad'] = SP.array([(info['grad']**2).sum()])

        return RV

    def fit_diagonal(self,XXstar=None,verbose=False):
        """ fit a diagonal model """

        if self.cache['diagonal']==None:
            #1. initialize params
            RV = self.fitSingleTraitModel(verbose=verbose)
            #2. init gp and params
            Cg = limix.CDiagonalCF(self.P)
            Cn = limix.CDiagonalCF(self.P)
            Cg_params0 = SP.sqrt(RV['varST'][:,0])
            Cn_params0 = SP.sqrt(RV['varST'][:,1])
            self.gp = gp2kronSum(self.Y,Cg,Cn,self.XX)
            params0 = {'Cg':Cg_params0,'Cn':Cn_params0}
            self.gp.setParams(params0)

            #4. output
            if XXstar!=None:
                RV['Ystar'] = self.gp.predict(XXstar)
            RV['Cg'] = Cg.K()
            RV['Cn'] = Cn.K()
            RV['conv'] = SP.array([True])
            self.cache['diagonal'] = copy.deepcopy(RV)
        else:
            RV = copy.deepcopy(self.cache['diagonal'])

        return RV

    def fit_freeformPen(self,lambd=0.5,XXstar=None,verbose=False):
        """
        fit a diagonal and a freeform model and mixes the covariance matrices
        for lambda=0: diagonal model, for lambda=1: freeform model 
        """
        RVd = self.fit_diagonal()
        RVf = self.fit_freeform()
        Cg = limix.CFixedCF(RVd['Cg']+lambd*(RVf['Cg']-RVd['Cg']))
        Cn = limix.CFixedCF(RVd['Cn']+lambd*(RVf['Cn']-RVd['Cn']))
        self.gp = gp2kronSum(self.Y,Cg,Cn,self.XX)
        params0 = {'Cg':SP.ones(1),'Cn':SP.ones(1)}
        self.gp.setParams(params0)
        #4. output
        RV = {}
        if XXstar!=None:
            RV['Ystar'] = self.gp.predict(XXstar)
        RV['Cg'] = Cg.K()
        RV['Cn'] = Cn.K()
        RV['conv'] = SP.array([RVf['conv'][0]*RVd['conv'][0]])

        return RV

    def crossValidation(self,seed=0,n_folds=10,verbose=True,model='freeform',rank=1,lambd=0.5):
        """ cross Validation """
        # split samples into training and test
        SP.random.seed(seed)
        r = SP.random.permutation(self.N)
        Icv = SP.floor(((SP.ones((self.N))*n_folds)*r)/self.Y.shape[0])

        Ystar = SP.zeros_like(self.Y)
        RV = {'Cg':SP.zeros((n_folds,self.P,self.P)),'Cn':SP.zeros((n_folds,self.P,self.P))}

        # loop over folds
        for fold_j in range(n_folds):

            if verbose:
                print ".. predict fold %d"%fold_j

            # split train and test
            Itrain  = Icv!=fold_j
            Itest   = Icv==fold_j
            Ytrain  = self.Y[Itrain,:]
            Ytest   = self.Y[Itest,:]
            XX      = self.XX[Itrain,:][:,Itrain]
            XXstar  = self.XX[Itest,:][:,Itrain]

            # fit model on train and predict on test
            vc = VarianceDecompositionPyGP(Ytrain,XX)
            _rv = vc.fit(model=model,lambd=lambd,rank=rank,XXstar=XXstar,verbose=True)
            assert _rv['conv'][0], 'VarianceDecompositon:: not converged for fold %d. Stopped here' % fold_j

            # stor prediction and covar matrives
            Ystar[Itest,:] = _rv['Ystar']
            RV['Cg'][fold_j,:,:] = _rv['Cg']
            RV['Cn'][fold_j,:,:] = _rv['Cn']

        return Ystar,RV
        
    def crossValidationLambd(self,seed=0,n_folds=10,verbose=True,lvs=None):
        """ cross validation on penalization """
        assert lvs!=None, 'specify lambda values'
        # split samples into training and test
        SP.random.seed(seed)
        r = SP.random.permutation(self.N)
        Icv = SP.floor(((SP.ones((self.N))*n_folds)*r)/self.Y.shape[0])

        Ystar = SP.zeros((lvs.shape[0],self.Y.shape[0],self.Y.shape[1]))
        RV = {'Cg':SP.zeros((lvs.shape[0],n_folds,self.P,self.P)),
                'Cn':SP.zeros((lvs.shape[0],n_folds,self.P,self.P))}

        # loop over folds
        for fold_j in range(n_folds):

            if verbose:
                print ".. predict fold %d"%fold_j

            # split train and test
            Itrain  = Icv!=fold_j
            Itest   = Icv==fold_j
            Ytrain  = self.Y[Itrain,:]
            Ytest   = self.Y[Itest,:]
            XX      = self.XX[Itrain,:][:,Itrain]
            XXstar  = self.XX[Itest,:][:,Itrain]

            # fit model on train and predict on test
            vc = VarianceDecompositionPyGP(Ytrain,XX)
            for lv_i in range(lvs.shape[0]):
                print "   .. lambda %.2f"%lvs[lv_i]
                _rv = vc.fit(model='freeformPen',lambd=lvs[lv_i],XXstar=XXstar,verbose=True)
                assert _rv['conv'][0], 'VarianceDecompositon:: not converged for fold %d. Stopped here' % fold_j
                # stor prediction and covar matrives
                Ystar[lv_i,Itest,:] = _rv['Ystar']
                RV['Cg'][lv_i,fold_j,:,:] = _rv['Cg']
                RV['Cn'][lv_i,fold_j,:,:] = _rv['Cn']

        return Ystar,RV
        
