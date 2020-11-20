import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from scipy import sparse
from timesteppers import *
from field import *
from spatial import *

class FokkerPlanck_1D:
    
    def __init__(self, X, mu, D, spatial_order=4):
      
        self.X = X
        p = self.X.field_list[0]
        self.domain = p.domain
        
        dx = FiniteDifferenceUniformGrid(1, spatial_order, p*mu)
        dx2 = FiniteDifferenceUniformGrid(2, spatial_order, p*D)
        
        pt = Field(self.domain)
        LHS = pt + dx - dx2
        
        bc1 =  Left(0, 4, p, 0)
        bc2 = Right(0, 4, p, 0)
        
        M = LHS.field_coeff(pt)
        M = M.tocsr()
        M[:1,:]  = bc1.field_coeff(pt)
        M[-1:,:] = bc2.field_coeff(pt)
        M.eliminate_zeros()
        self.M = M
        
        L = LHS.field_coeff(p)
        L = L.tocsr()
        L[:1,:]  = bc1.field_coeff(p)
        L[-1:,:] = bc2.field_coeff(p)
        L.eliminate_zeros()
        self.L = L
        
        self.ts = CrankNicolson(self, axis=0)
                    
        self.t = 0.
        self.iter = 0
        
    def step(self, dt):

        self.ts.step(dt)
        self.t += dt
        self.iter += 1
        
class FokkerPlanck_2D:
    
    def __init__(self, X,mu_i,D_ij, spatial_order=4):
      
     
        self.X = X
        p = X.field_list[0]        
        self.domain = p.domain
        
        pt = Field(self.domain)
                             
        dx_pmu = FiniteDifferenceUniformGrid(1, spatial_order, mu_i[0]*p, axis=0)
        dy_pmu = FiniteDifferenceUniformGrid(1, spatial_order, mu_i[1]*p, axis=1)
        
        dxx_pD = FiniteDifferenceUniformGrid(2, spatial_order, D_ij[0][0]*p,   axis = 0)
        dyy_pD = FiniteDifferenceUniformGrid(2, spatial_order, D_ij[1][1]*p,   axis = 1)
        dx_pD  = FiniteDifferenceUniformGrid(1, spatial_order, D_ij[1][0]*p,   axis = 0)
        dxy_pD = FiniteDifferenceUniformGrid(1, spatial_order, dx_pD, axis = 1)
        dy_pD  = FiniteDifferenceUniformGrid(1, spatial_order, D_ij[0][1]*p,   axis = 1)
        dyx_pD = FiniteDifferenceUniformGrid(1, spatial_order, dy_pD, axis = 0)
        
        LHS = pt  + dx_pmu + dy_pmu - dxx_pD - dyy_pD - dxy_pD - dyx_pD


        self.bc1x =  Left(0, spatial_order, p, 0, axis=0)
        self.bc2x = Right(0, spatial_order, p, 0, axis=0)
        self.bc1y =  Left(0, spatial_order, p, 0,axis=1)
        self.bc2y = Right(0, spatial_order, p, 0,axis=1)
        
        
        M = LHS.field_coeff(pt,axis='full')
        self.M = M
        
        L = LHS.field_coeff(p,axis='full')
        self.L = L
        
        
        self.t = 0.
        self.iter = 0
        
        
    def step(self, dt):
        
        if 1:
            self.bc1x.operate()
            self.bc2x.operate()
            self.bc1y.operate()
            self.bc2y.operate()

        self.ts = CrankNicolson3(self,axis='full')
        self.ts.step(dt)
        self.t += dt
        self.iter += 1