
#  Fulya Kiroglu - Astronomy 
#  October 27,2020 
#  ESAM 446-1 
#  Problem set 6 - first version
# Solve the ReactionDiffusion2D and ViscousBurgers2D equations using the operator splitting approach


from field import Field, FieldSystem
from scipy import sparse
import field
import spatial
import timesteppers
from timesteppers import CrankNicolson, PredictorCorrector
from spatial import FiniteDifferenceUniformGrid

class Reaction:
    
    def __init__(self, X):
        
        self.X = X
        c = X.field_list[0]
        self.domain = c.domain
        
        self.F_ops = [c-c*c]
        
class Diffusionx:
    
    def __init__(self, X, D, dcdx2):
        c = X.field_list[0]
        self.X = X
        self.domain = c.domain
        
        ct = field.Field(self.domain)
        
        eq1 = ct - D*dcdx2
        
        self.M = eq1.field_coeff(ct, axis=0)
        self.L = eq1.field_coeff(c, axis=0)
        
class Diffusiony:
    
    def __init__(self, X, D, dcdy2):
        c = X.field_list[0]
        self.X = X
        self.domain = c.domain
        
        ct = field.Field(self.domain)
        
        eq1 = ct - D*dcdy2
        
        self.M = eq1.field_coeff(ct, axis=1)
        self.L = eq1.field_coeff(c, axis=1)

class ReactionDiffusion2D:

    def __init__(self, X, D, dcdx2, dcdy2):
        
        self.iter=0
        self.t=0
        self.dt=None

        self.diffx = Diffusionx(X, D, dcdx2)
        
        self.diffy = Diffusiony(X, D, dcdy2)
        
        self.react = Reaction(X)


    def step(self, dt):
        
        #for the diffusion terms, use the CrankNicolson scheme
        ts_x = timesteppers.CrankNicolson(self.diffx, 0)
        ts_y = timesteppers.CrankNicolson(self.diffy, 1)
        #for the reaction term use the PredictorCorrector scheme
        ts = timesteppers.PredictorCorrector(self.react)
        
        if self.dt != dt:
            # take a step
            ts.step(dt/2)
            ts_y.step(dt)
            ts_x.step(dt)
            ts.step(dt/2)
            
            # update self.t and self.iter
            self.t += dt
            self.iter += 1
            
class Viscousx:
    
    def __init__(self, X, nu, dudx2,dvdx2):
        
        # defines dt u - nu dudx2 =0 and
        # dt v - nu dvdx2=0
        
        u = X.field_list[0]
        v = X.field_list[1]
        self.X = X
        
        self.domain = u.domain
        ut = field.Field(self.domain)
        
        eq1 = ut - nu*dudx2
        
        vt = field.Field(self.domain)
        
        eq2 = vt- nu*dvdx2
        
        
        M00 = eq1.field_coeff(ut,axis=0)
        M01 = eq1.field_coeff(vt,axis=0)
        M10 = eq2.field_coeff(ut,axis=0)
        M11 = eq2.field_coeff(vt,axis=0)
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = eq1.field_coeff(u,axis=0)
        L01 = eq1.field_coeff(v,axis=0)
        L10 = eq2.field_coeff(u,axis=0)
        L11 = eq2.field_coeff(v,axis=0)
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])
        
        
class Viscousy:
    
    def __init__(self, X, nu, dudy2,dvdy2):
        
        #define dt u - nu dudy2 =0 and
        #dt v - nu dvdy2=0
        
        u = X.field_list[0]
        v = X.field_list[1]
        self.X = X
        
        self.domain = u.domain
        ut = field.Field(self.domain)
        
        eq1 = ut - nu*dudy2
        
        vt = field.Field(self.domain)
        
        eq2 = vt- nu*dvdy2
        
        
        M00 = eq1.field_coeff(ut,axis=1)
        M01 = eq1.field_coeff(vt,axis=1)
        M10 = eq2.field_coeff(ut,axis=1)
        M11 = eq2.field_coeff(vt,axis=1)
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = eq1.field_coeff(u,axis=1)
        L01 = eq1.field_coeff(v,axis=1)
        L10 = eq2.field_coeff(u,axis=1)
        L11 = eq2.field_coeff(v,axis=1)
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])
        
class Burgers:
    
      def __init__(self, X, dudx, dudy,dvdx,dvdy):
            
        # define dt u + u * dudx + v * dudy = 0 and
        # dt v + u * dvdx + v * dvdy= 0
        
        u = X.field_list[0]
        v = X.field_list[1]
        self.X = X
        
        self.domain = u.domain
        
        self.F_ops = [-u*dudx - v*dudy, -u*dvdx - v*dvdy]


class ViscousBurgers2D:

    def __init__(self, X, nu, spatial_order):
        
        self.iter=0
        self.t=0
        self.dt=None
        
        u = X.field_list[0]
        v = X.field_list[1]
        self.X = X
        
        self.domain = u.domain
        
        dudx2 = spatial.FiniteDifferenceUniformGrid(2, spatial_order, u, 0)
        dudy2 = spatial.FiniteDifferenceUniformGrid(2, spatial_order, u, 1)
        dvdx2 = spatial.FiniteDifferenceUniformGrid(2, spatial_order, v, 0)
        dvdy2 = spatial.FiniteDifferenceUniformGrid(2, spatial_order, v, 1)
        
        dudx = spatial.FiniteDifferenceUniformGrid(1, spatial_order, u, 0)
        dudy= spatial.FiniteDifferenceUniformGrid(1, spatial_order, u, 1)
        dvdx= spatial.FiniteDifferenceUniformGrid(1, spatial_order, v, 0)
        dvdy = spatial.FiniteDifferenceUniformGrid(1, spatial_order, v, 1)
        
        viscx = Viscousx(X, nu, dudx2,dvdx2)
        
        viscy = Viscousy(X, nu, dudy2,dvdy2)

        burgers = Burgers(X, dudx, dudy,dvdx,dvdy)
        
        self.ts = timesteppers.PredictorCorrector(burgers)
        
        self.ts_x = timesteppers.CrankNicolson(viscx, 0)
        self.ts_y = timesteppers.CrankNicolson(viscy, 1)
    
        
        
    def step(self, dt):
        
        if self.dt != dt:
            # take a step
            self.ts.step(dt/2)
            self.ts_y.step(dt)
            self.ts_x.step(dt)
            self.ts.step(dt/2)

            # update self.t and self.iter
            self.t += dt
            self.iter += 1

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


