#  Fulya Kiroglu - Astronomy 
#  October 20,2020 
#  ESAM 446-1 
#  Problem set 5 - first version

# Implement Equations(Sound wave & Reaction-Diffusion)

from field import Field, FieldSystem
from scipy import sparse

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
        
        
        self.grid = u.grid
        self.X = FieldSystem([u, p])
        
        ut = Field(self.grid)
        pt = Field(self.grid)
        
        eq1= rho0 * ut + dp
        eq2= pt + gamma_p0 * du
        
        M00 = eq1.field_coeff(ut)
        M01 = eq1.field_coeff(pt)
        M10 = eq2.field_coeff(ut)
        M11 = eq2.field_coeff(pt)
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])
        
        L00 = eq1.field_coeff(u)
        L01 = eq1.field_coeff(p)
        L10 = eq2.field_coeff(u)
        L11 = eq2.field_coeff(p)
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])
        
        self.F_ops = [0*u, 0*u]


class ReactionDiffusion:
    
    def __init__(self, c, d2c, c_target, D):
        
        self.grid = c.grid
        
        self.X = FieldSystem([c])
        
        ct = Field(self.grid)
        
        eq1= ct - D * d2c
        
        self.M = eq1.field_coeff(ct)
    
        self.L = eq1.field_coeff(c)
        
        self.F_ops = [c_target*c-c*c]


