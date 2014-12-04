import sys
import os
import unittest
import scipy as SP
import pdb
from write import *

class unitestClass(unittest.TestCase):
    """class for unitests """ 
    
    def setUp(self):
        pass

    def loadData(self):
        self.Y = SP.loadtxt('./data/Y.txt') 
        self.XX = SP.loadtxt('./data/XX.txt') 
        self.Xr = SP.loadtxt('./data/Xr.txt') 
        self.N,self.P = self.Y.shape
        self.write = write 

    def saveStuff(self,test,ext):
        """ util function """ 
        base = './data/res_'+self.module+'_'+test+'_'
        for key in ext.keys(): 
            SP.savetxt(base+key+'.txt',ext[key])

    def loadStuff(self,test,keys):
        """ util function """ 
        RV = {}
        base = './data/res_'+self.module+'_'+test+'_'
        for key in keys: 
            RV[key] = SP.loadtxt(base+key+'.txt')
        return RV

    def assess(self,test,ext):
        """ returns a bool vector """
        real = self.loadStuff(test,ext.keys()) 
        RV = SP.all([((ext[key]-real[key])**2).mean()<1e-6 for key in ext.keys()])
        return RV

