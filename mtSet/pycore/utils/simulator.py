"""
Created on Sep 24, 2013
@author: casale
"""
import scipy as SP
import scipy.linalg as LA
import pdb

import sys
sys.path.append('./../../..')
import mtSet.pycore.external.limix.plink_reader as plink_reader


def genBinormal(dim1,dim2,percSign=0.5,std=1e-1):
    rv = (2*(SP.rand(dim1,dim2)>percSign)-1)+std*SP.randn(dim1,dim2)
    return rv

def selectRnd(n_sel,n_all):
    rv = SP.zeros(n_all)
    rv[:n_sel] = 1
    SP.random.shuffle(rv)
    rv = rv==1
    return rv

class CSimulator:
    """
    this class takes care of phenotype generation in a flexible way
    """

    def __init__(self,bfile,XX=None,P=1):
        """
        X: genotype matrix
        traitNum: number of traits to be considered
        """
        self.bfile = bfile
        self.N = XX.shape[0]
        self.P = P
        self.XX = XX

        pass
    
    def getRegion(self,size=3e4,min_nSNPs=1,chrom_i=None,pos_min=None,pos_max=None):
        """
        Sample a region from the piece of genotype X, chrom, pos
        minSNPnum:  minimum number of SNPs contained in the region
        Ichrom:  restrict X to chromosome Ichrom before taking the region
        cis:        bool vector that marks the sorted region
        region:  vector that contains chrom and init and final position of the region
        """
        bim = plink_reader.readBIM(self.bfile,usecols=(0,1,2,3))
        
        chrom = SP.array(bim[:,0],dtype=int)
        pos   = SP.array(bim[:,3],dtype=int)
        
        if chrom_i is None:
            n_chroms = chrom.max()
            chrom_i  = int(SP.ceil(SP.rand()*n_chroms))

        pos   = pos[chrom==chrom_i]
        chrom = chrom[chrom==chrom_i]

        ipos = SP.ones(len(pos),dtype=bool)
        if pos_min is not None:
            ipos = SP.logical_and(ipos,pos_min<pos)

        if pos_max is not None:
            ipos = SP.logical_and(ipos,pos<pos_max)

        pos = pos[ipos]
        chrom = chrom[ipos]
        
        if size==1:
            # select single SNP
            idx = int(SP.ceil(pos.shape[0]*SP.rand()))
            cis  = SP.arange(pos.shape[0])==idx
            region = SP.array([chrom_i,pos[idx],pos[idx]])
        else:
            while 1:
                idx = int(SP.floor(pos.shape[0]*SP.rand()))
                posT1 = pos[idx]
                posT2 = pos[idx]+size
                if posT2<=pos.max():
                    cis = chrom==chrom_i
                    cis*= (pos>posT1)*(pos<posT2)
                    if cis.sum()>min_nSNPs: break
            region = SP.array([chrom_i,posT1,posT2])

        start = SP.nonzero(cis)[0].min()
        nSNPs  = cis.sum()
        rv = plink_reader.readBED(self.bfile,useMAFencoding=True,start = start, nSNPs = nSNPs,bim=bim)
        Xr = rv['snps']
        return Xr, region


    def genRegionTerm(self,X,vTot=0.1,pCausal=0.10,nCausal=None,pCommon=1.,nCommon=None,plot=False,distribution='biNormal'):
        """
        Generate population structure term
        Population structure is simulated by background SNPs
        
        beta_pdf:        pdf used to generate the regression weights
                          for now either Normal or fixed
        variance:        variance of the term
        percCausal:    percentage of causal SNPs
        Xcausal:          set of SNPs being causal
        """
        S = X.shape[1]

        # number of causal, common, specific
        if nCausal==None:
            nCausal=int(SP.floor(pCausal*S))
        if nCommon==None:
            nCommon = round(pCommon*nCausal)
        nSpecific = self.P*(nCausal-nCommon)

        # common SNPs
        if nCommon>0:
            if distribution=='biNormal':
                Bc  = SP.kron(genBinormal(nCommon,1),genBinormal(1,self.P))
            elif distribution=='normal':
                Bc  = SP.kron(SP.randn(nCommon,1),SP.randn(1,self.P))
            Ic  = selectRnd(nCommon,S)
            Yc  = SP.dot(X[:,Ic],Bc)
            Yc *= SP.sqrt(nCommon/Yc.var(0).mean())
        else:
        
            Yc = SP.zeros((self.N,self.P))
        
        
        
        # indipendent signal
        if nSpecific>0:
            Is  = selectRnd(nSpecific,S*self.P).reshape(S,self.P)
            if distribution=='biNormal':
                Bi  = Is*genBinormal(S,self.P)
            elif distribution=='normal':
                Bi  = Is*SP.randn(S,self.P)
            Yi  = SP.dot(X,Bi)
            Yi *= SP.sqrt(nSpecific/(Yi.var(0).mean()*self.P))
        else:
            Yi = SP.zeros((self.N,self.P))
        
        
        Y   = Yc+Yi 
        Yc *= SP.sqrt(vTot/Y.var(0).mean())
        Yi *= SP.sqrt(vTot/Y.var(0).mean())

        if plot:
            import pylab as PL
            PL.ion()
            for p in range(self.P):
                PL.subplot(self.P,1,p+1)
                PL.plot(SP.arange(S)[Ic],Bc[:,p],'o',color='y')
                _Is = Is[:,p]
                if _Is.sum()>0:
                    PL.plot(SP.arange(S)[_Is],Bi[_Is,p],'o',color='r') 
                #PL.ylim(-2,2)
                PL.plot([0,S],[0,0],'k')

        return Yc, Yi


    def _genBgTerm_fromSNPs(self,vTot=0.5,vCommon=0.1,pCausal=0.5,plot=False):
        """ generate  """

        print 'Reading in all SNPs. This is slow.'
        rv = plink_reader.readBED(self.bfile,useMAFencoding=True)
        X  = rv['snps']
        S  = X.shape[1]
        vSpecific = vTot-vCommon

        # select causal SNPs
        nCausal = int(SP.floor(pCausal*S))
        Ic = selectRnd(nCausal,S)
        X = X[:,Ic]

        # common effect
        Bc  = SP.kron(SP.randn(nCausal,1),SP.randn(1,self.P))
        Yc  = SP.dot(X,Bc)
        Yc *= SP.sqrt(vCommon/Yc.var(0).mean())

        # indipendent effect
        Bi  = SP.randn(nCausal,self.P)
        Yi  = SP.dot(X,Bi)
        Yi *= SP.sqrt(vSpecific/Yi.var(0).mean())

        if plot:
            import pylab as PL
            PL.ion()
            for p in range(self.P):
                PL.subplot(self.P,1,p+1)
                PL.plot(SP.arange(self.X.shape[1])[Ic],Bc[:,p],'o',color='y',alpha=0.05)
                PL.plot(SP.arange(self.X.shape[1])[Ic],Bi[:,p],'o',color='r',alpha=0.05) 
                #PL.ylim(-2,2)
                PL.plot([0,Ic.shape[0]],[0,0],'k')

        return Yc, Yi

    def _genBgTerm_fromXX(self,vTot,vCommon,XX,a=None,c=None):
        """
        generate background term from SNPs

        Args:
            vTot: variance of Yc+Yi
            vCommon: variance of Yc
            XX: kinship matrix
            a: common scales, it can be set for debugging purposes
            c: indipendent scales, it can be set for debugging purposes
        """
        vSpecific = vTot-vCommon

        if a==None: a = SP.randn(self.P)
        if c==None: c = SP.randn(self.P)


        XX += 1e-3 * SP.eye(XX.shape[0])
        L = LA.cholesky(XX,lower=True)
        # common effect
        R = SP.randn(self.N,self.P)
        A = SP.zeros((self.P,self.P))
        A[:,0] = a
        Yc = SP.dot(L,SP.dot(R,A.T))
        Yc*= SP.sqrt(vCommon)/SP.sqrt(Yc.var(0).mean())
        # specific effect
        R = SP.randn(self.N,self.P)
        Yi = SP.dot(L,SP.dot(R,SP.diag(c)))
        Yi*= SP.sqrt(vSpecific)/SP.sqrt(Yi.var(0).mean())

        return Yc, Yi
        

    def genBgTerm(self,vTot=0.5,vCommon=0.1,pCausal=0.5,XX=None,use_XX=False,a=None,c=None,plot=False):
        """ generate  """
        if use_XX:
            if XX is None:  XX = self.XX
            assert XX is not None, 'Simulator: set XX!'
            Yc,Yi = self._genBgTerm_fromXX(vTot,vCommon,XX,a=a,c=c)
        else:
            Yc,Yi = self._genBgTerm_fromSNPs(vTot=vTot,vCommon=vCommon,pCausal=pCausal,plot=plot)
        return Yc, Yi

    def genHidden(self,nHidden=10,vTot=0.5,vCommon=0.1):
        """ generate  """

        vSpecific = vTot-vCommon

        # generate hidden
        X = SP.randn(self.N,nHidden)

        # common effect
        Bc  = SP.kron(SP.randn(nHidden,1),SP.randn(1,self.P))
        Yc  = SP.dot(X,Bc)
        Yc *= SP.sqrt(vCommon/Yc.var(0).mean())
        
        # indipendent effect
        Bi  = SP.randn(nHidden,self.P)
        Yi  = SP.dot(X,Bi)
        Yi *= SP.sqrt(vSpecific/Yi.var(0).mean())

        return Yc,Yi

    def genNoise(self,vTot=0.4,vCommon=0.2):

        vSpecifc = vTot-vCommon

        # common
        Yc  = SP.kron(SP.randn(self.N,1),SP.randn(1,self.P))
        Yc *= SP.sqrt(vCommon/Yc.var(0).mean())

        # independent
        Yi  = SP.randn(self.N,self.P)
        Yi *= SP.sqrt(vSpecifc/Yi.var(0).mean())

        return Yc,Yi


    def genPheno(self,Xr,
        vTotR=0.1,nCommonR=5,nCausalR=10,distribution='biNormal',
        vCommonBg=0.1,vTotBg=0.4,pCausalBg=0.5,XX=None,use_XX=False,
        vCommonH=0.1,vTotH=0.2,nHidden=10,
        vCommonN=0.,vTotN=0.3,standardize=True):

      
        YRc,YRi = self.genRegionTerm(Xr,vTot=vTotR,nCommon=nCommonR,nCausal=nCausalR,distribution='biNormal')
        YGc,YGi = self.genBgTerm(vCommon=vCommonBg,vTot=vTotBg,pCausal=pCausalBg,XX=XX,use_XX=use_XX)
        YHc,YHi = self.genHidden(vCommon=vCommonH,vTot=vTotH,nHidden=nHidden)
        YNc,YNi = self.genNoise(vCommon=vCommonN,vTot=vTotN)        
        Y = YRc+YRi+YGc+YGi+YHc+YHi+YNc+YNi

        if standardize:
            Y -= Y.mean(0)
            Y /= Y.std(0)

        info = {'YRc':YRc,'YRi':YRi,
                'YGc':YGc,'YGi':YGi,
                'YHc':YHc,'YHi':YHi,
                'YNc':YNc,'YNi':YNi}

        return Y, info

