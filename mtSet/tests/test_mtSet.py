import sys
import os
from unitestClass import unitestClass
path_abs = os.path.dirname(os.path.abspath(sys.argv[0]))
path_mtSet = os.path.join(path_abs,'../..')
sys.path.append(path_mtSet)
import mtSet.pycore.modules.multiTraitSetTest as MTST
import unittest
import scipy as SP
import scipy.linalg as LA
import pdb

class mtSet_test(unitestClass):
    """test class for gp2kronSum"""
    
    def setUp(self):
        SP.random.seed(0)
        self.module = 'mtSet'
        self.loadData() 

    def test_base(self):
        test = 'base'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=self.XX)
        optInfo = setTest.optimize(self.Xr)
        ext = {'Cr':optInfo['Cr'],
               'Cg':optInfo['Cg'],
               'Cn':optInfo['Cn']}
        if self.write: self.saveStuff(test,ext)
        RV = self.assess(test,ext)
        self.assertTrue(RV)

    def test_fixed(self):
        test = 'fixed'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=self.XX,F=self.Xr[:,:2])
        optInfo = setTest.optimize(self.Xr)
        ext = {'Cr':optInfo['Cr'],
               'Cg':optInfo['Cg'],
               'Cn':optInfo['Cn']}
        if self.write: self.saveStuff(test,ext)
        RV = self.assess(test,ext)
        self.assertTrue(RV)

    def test_eigenCache(self):
        test = 'eigenCache'
        S,U = LA.eigh(self.XX)
        setTest = MTST.MultiTraitSetTest(self.Y,S_XX=S,U_XX=U)
        optInfo = setTest.optimize(self.Xr)
        ext = {'Cr':optInfo['Cr'],
               'Cg':optInfo['Cg'],
               'Cn':optInfo['Cn']}
        if self.write: self.saveStuff(test,ext)
        RV = self.assess(test,ext)
        self.assertTrue(RV)

if __name__ == '__main__':
    unittest.main()


