
from field import Field, FieldSystem
from scipy import sparse
from timesteppers import CrankNicolson, PredictorCorrector
from spatial import FiniteDifferenceUniformGrid

class ViscousBurgers2D:

    def __init__(self, X, nu, spatial_order):
        # initialized
        pass

    def step(self, dt):
        # take a step
        # update self.t and self.iter
        pass


class ViscousBurgers:
    
    def __init__(self, u, nu, du, d2u):
        self.grid = u.grid
        self.X = FieldSystem([u])

        ut = Field(self.grid)
                
        LHS = ut - nu*d2u
        
        self.M = LHS.field_coeff(ut)
        self.L = LHS.field_coeff(u)
        
        self.F_ops = [-u*du]


class Wave:
    
    def __init__(self, u, v, d2u):
        self.grid = u.grid
        self.X = FieldSystem([u, v])

        ut = Field(self.grid)
        vt = Field(self.grid)

        eq1 = ut - v
        eq2 = vt - d2u

        M00 = eq1.field_coeff(ut)
        M01 = eq1.field_coeff(vt)
        M10 = eq2.field_coeff(ut)
        M11 = eq2.field_coeff(vt)
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = eq1.field_coeff(u)
        L01 = eq1.field_coeff(v)
        L10 = eq2.field_coeff(u)
        L11 = eq2.field_coeff(v)
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        self.F_ops = [0*u, 0*v]


class SoundWave:

    def __init__(self, u, p, du, dp, rho0, gamma_p0):
        pass


class ReactionDiffusion:
    
    def __init__(self, c, d2c, c_target, D):
        pass

    
class Diffusion_split:
    
    def __init__(self, X, D, dc2, axis):
        c = X.field_list[0]
        self.X = X
        self.domain = c.domain
        
        ct = Field(self.domain)
        
        eq1 = ct - D*dc2
        
        self.M = eq1.field_coeff(ct, axis=axis)
        self.L = eq1.field_coeff(c, axis=axis)
        self.F_ops = [0*c]
        
class Reaction:
    def __init__(self, X):
        c = X.field_list[0]
        self.X = X
        self.domain = c.domain
        
        ct = Field(self.domain)
        
        eq1 = ct
        
        self.M = eq1.field_coeff(ct)
        self.L = eq1.field_coeff(c)
        self.F_ops = [c*(1-c)]


class ReactionDiffusion2D:

    def __init__(self, X, D, dcdx2, dcdy2):
        ## init
        self.t    = 0.
        self.iter = 0
        
        c = X.field_list[0]
        self.X = X
        self.domain = c.domain
                
        self.operators = [Diffusion_split(self.X, D, dcdx2, 0), Diffusion_split(self.X, D, dcdy2, 1), Reaction(self.X)]
        self.ts_list   = [CrankNicolson(self.operators[0], 0),
                          CrankNicolson(self.operators[1], 1),
                          PredictorCorrector(self.operators[2])]        

    def step(self, dt):
        
        # diffusive terms commute, only have to strang split the other terms
        self.ts_list[2].step(dt/2.)
        self.ts_list[1].step(dt)
        self.ts_list[0].step(dt)
        self.ts_list[2].step(dt/2.)
            
        
        # update self.t and self.iter
        self.t += dt
        self.iter += 1

       
class VB_diffusion_split:
    
    def __init__(self, X, nu, du2, dv2, axis):
        u = X.field_list[0]
        v = X.field_list[1]
        self.X = X        
        self.domain = u.domain
        
        ut = Field(self.domain)
        vt = Field(self.domain)
        
        eq1_LHS = ut - nu*du2
        eq2_LHS = vt - nu*dv2
        
        M00 = eq1_LHS.field_coeff(ut, axis=axis)
        M01 = eq1_LHS.field_coeff(vt, axis=axis)
        M10 = eq2_LHS.field_coeff(ut, axis=axis)
        M11 = eq2_LHS.field_coeff(vt, axis=axis)
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = eq1_LHS.field_coeff(u, axis=axis)
        L01 = eq1_LHS.field_coeff(v, axis=axis)
        L10 = eq2_LHS.field_coeff(u, axis=axis)
        L11 = eq2_LHS.field_coeff(v, axis=axis)
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])        
        
        self.F_ops = [0*u, 0*v]
        
class VB_advection_split:
    
    def __init__(self, X, dudx, dudy, dvdx, dvdy):
        u = X.field_list[0]
        v = X.field_list[1]
        self.X = X        
        self.domain = u.domain

        ut = Field(self.domain)
        vt = Field(self.domain)
        
        eq1_LHS = ut
        eq2_LHS = vt
        
        M00 = eq1_LHS.field_coeff(ut)
        M01 = eq1_LHS.field_coeff(vt)
        M10 = eq2_LHS.field_coeff(ut)
        M11 = eq2_LHS.field_coeff(vt)
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = eq1_LHS.field_coeff(u)
        L01 = eq1_LHS.field_coeff(v)
        L10 = eq2_LHS.field_coeff(u)
        L11 = eq2_LHS.field_coeff(v)
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        self.F_ops = [-u*dudx - v*dudy, -u*dvdx - v*dvdy]


class ViscousBurgers2D:

    def __init__(self, X, nu, spatial_order):
        ## init
        self.t    = 0.
        self.iter = 0
        
        ## init domain
        u = X.field_list[0]
        v = X.field_list[1]
        self.X = X
        self.domain = u.domain

        ## init derivative operators
        dudx  = FiniteDifferenceUniformGrid(1, spatial_order, u, 0)
        dudy  = FiniteDifferenceUniformGrid(1, spatial_order, u, 1)
        dvdx  = FiniteDifferenceUniformGrid(1, spatial_order, v, 0)
        dvdy  = FiniteDifferenceUniformGrid(1, spatial_order, v, 1)
        dudx2 = FiniteDifferenceUniformGrid(2, spatial_order, u, 0)
        dudy2 = FiniteDifferenceUniformGrid(2, spatial_order, u, 1)
        dvdx2 = FiniteDifferenceUniformGrid(2, spatial_order, v, 0)
        dvdy2 = FiniteDifferenceUniformGrid(2, spatial_order, v, 1)
        
        
        
        self.operators = [VB_diffusion_split(self.X, nu, dudx2, dvdx2,0),
                          VB_diffusion_split(self.X, nu, dudy2, dvdy2,1),
                          VB_advection_split(self.X, dudx, dudy, dvdx, dvdy)]
            
            
        self.ts_list   = [CrankNicolson(self.operators[0], 0),
                          CrankNicolson(self.operators[1], 1),
                          PredictorCorrector(self.operators[2])]        

    def step(self, dt):
        # take a step
        # update self.t and self.iter
        
        # diffusive terms commute, only have to strang split the other terms
        self.ts_list[2].step(dt/2.)
        self.ts_list[1].step(dt)
        self.ts_list[0].step(dt)
        self.ts_list[2].step(dt/2.)
            
        
        # update self.t and self.iter
        self.t += dt
        self.iter += 1
