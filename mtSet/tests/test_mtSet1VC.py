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

class mtSet1VC_test(unitestClass):
    """test class for gp2kronSum"""
    
    def setUp(self):
        SP.random.seed(0)
        self.module = 'mtSet1VC'
        self.loadData() 

    def test_base(self):
        test = 'base'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=None)
        optInfo = setTest.optimize(self.Xr)
        ext = {'Cr':optInfo['Cr'],
               'Cn':optInfo['Cn']}
        if self.write: self.saveStuff(test,ext)
        RV = self.assess(test,ext)
        self.assertTrue(RV)

    def test_fixed(self):
        test = 'fixed'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=None,F=self.Xr[:,:2])
        optInfo = setTest.optimize(self.Xr)
        ext = {'Cr':optInfo['Cr'],
               'Cn':optInfo['Cn']}

        if self.write: self.saveStuff(test,ext)
        RV = self.assess(test,ext)
        self.assertTrue(RV)

if __name__ == '__main__':
    unittest.main()


