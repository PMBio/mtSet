import sys
import os
path_abs = os.path.dirname(os.path.abspath(sys.argv[0]))
path_mtSet = os.path.join(path_abs,'../..')
sys.path.append(path_mtSet)
import mtSet.pycore.modules.multiTraitSetTest as MTST
import unittest
import scipy as SP
import scipy.linalg as LA
import pdb

class unitestClass(unittest.TestCase):
    """test class for optimization""" 
    
    def setUp(self):
        SP.random.seed(0)
        self.Y = SP.loadtxt('./data/Y.txt') 
        self.XX = SP.loadtxt('./data/XX.txt') 
        self.Xr = SP.loadtxt('./data/Xr.txt') 
        self.N,self.P = self.Y.shape
        self.write = False 

    def test_mtSetNull_base(self):
        fbasename = 'mtSetNull_base'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=self.XX)
        nullMTInfo = setTest.fitNull(cache=False)
        ext = {'Cg':nullMTInfo['Cg'],'Cn':nullMTInfo['Cn']}
        if self.write: self.saveStuff(fbasename,ext)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def test_mtSetNull_fixed(self):
        fbasename = 'mtSetNull_fixed'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=self.XX,F=self.Xr)
        nullMTInfo = setTest.fitNull(cache=False)
        ext = {'Cg':nullMTInfo['Cg'],'Cn':nullMTInfo['Cn'],
                'weights':nullMTInfo['params_mean']}
        if self.write: self.saveStuff(fbasename,ext)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def test_mtSetNull_eigenCache(self):
        fbasename = 'mtSetNull_base'
        S,U = LA.eigh(self.XX)
        setTest = MTST.MultiTraitSetTest(self.Y,S_XX=S,U_XX=U)
        nullMTInfo = setTest.fitNull(cache=False)
        ext = {'Cg':nullMTInfo['Cg'],'Cn':nullMTInfo['Cn']}
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def test_mtSet_base(self):
        fbasename = 'mtSet_base'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=self.XX)
        optInfo = setTest.optimize(self.Xr)
        ext = {'Cr':optInfo['Cr'],
               'Cg':optInfo['Cg'],
               'Cn':optInfo['Cn']}
        if self.write: self.saveStuff(fbasename)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def test_mtSet_fixed(self):
        fbasename = 'mtSet_fixed'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=self.XX,F=self.Xr[:,:2])
        optInfo = setTest.optimize(self.Xr)
        ext = {'Cr':optInfo['Cr'],
               'Cg':optInfo['Cg'],
               'Cn':optInfo['Cn']}
        if self.write: self.saveStuff(fbasename)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def test_mtSet_eigenCache(self):
        fbasename = 'mtSet_base'
        S,U = LA.eigh(self.XX)
        setTest = MTST.MultiTraitSetTest(self.Y,S_XX=S,U_XX=U)
        optInfo = setTest.optimize(self.Xr)
        ext = {'Cr':optInfo['Cr'],
               'Cg':optInfo['Cg'],
               'Cn':optInfo['Cn']}
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def test_mtSet1VCnull_base(self):
        fbasename = 'mtSet1VCnull_base'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=None)
        nullMTInfo = setTest.fitNull(cache=False)
        ext = {'Cg':nullMTInfo['Cg'],'Cn':nullMTInfo['Cn']}
        if self.write: self.saveStuff(fbasename,ext)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def test_mtSet1VCnull_fixed(self):
        fbasename = 'mtSet1VCnull_fixed'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=None,F=self.Xr)
        nullMTInfo = setTest.fitNull(cache=False)
        ext = {'Cg':nullMTInfo['Cg'],'Cn':nullMTInfo['Cn'],
                'weights':nullMTInfo['params_mean']}
        if self.write: self.saveStuff(fbasename,ext)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def test_mtSet1VC_base(self):
        fbasename = 'mtSet1VC_base'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=None)
        optInfo = setTest.optimize(self.Xr)
        ext = {'Cr':optInfo['Cr'],
               'Cn':optInfo['Cn']}
        if self.write: self.saveStuff(fbasename,ext)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def test_mtSet1VC_fixed(self):
        fbasename = 'mtSet1VC_fixed'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=None,F=self.Xr[:,:2])
        optInfo = setTest.optimize(self.Xr)
        ext = {'Cr':optInfo['Cr'],
               'Cn':optInfo['Cn']}
        if self.write: self.saveStuff(fbasename,ext)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def saveStuff(self,fbasename,ext):
        """ util function """ 
        base = './data/res_'+fbasename+'_'
        for key in ext.keys(): 
            SP.savetxt(base+key+'.txt',ext[key])

    def loadStuff(self,fbasename,keys):
        """ util function """ 
        RV = {}
        base = './data/res_'+fbasename+'_'
        for key in keys: 
            RV[key] = SP.loadtxt(base+key+'.txt')
        return RV

    def assess(self,fbasename,ext):
        """ returns a bool vector """
        real = self.loadStuff(fbasename,ext.keys()) 
        RV = SP.all([((ext[key]-real[key])**2).mean()<1e-6 for key in ext.keys()])
        return RV

if __name__ == '__main__':

    # Gather all tests in suite
    tests = unittest.TestLoader().discover('.','run_test.py')
    suite = unittest.TestSuite(tests)

    # run all tests
    unittest.TextTestRunner(verbosity=2).run(suite)

