#  Fulya Kiroglu - Astronomy 
#  October 21,2020 
#  ESAM 446-1 
#  Problem set 5 - second version
# Implement IMEX Timesteppers(BDFExtrapolate)

import numpy as np
from field import Field, FieldSystem, Identity, Average3
from scipy.special import factorial
import scipy.sparse.linalg as spla
from collections import deque

class IMEXTimestepper:

    def __init__(self, eq_set):
        self.t = 0
        self.iter = 0
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F_ops = eq_set.F_ops
        X_rhs = []
        for field in self.X.field_list:
            X_rhs.append(Field(field.grid))
        self.F = FieldSystem(X_rhs)
        self.X_rhs = X_rhs
        self.dt = None

    def evolve(self, time, dt):
        while self.t < time - 1e-8:
            self.step(dt)

    def step(self, dt):
        self._step(dt)
        self.t += dt
        self.iter += 1


class Euler(IMEXTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt*self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.dt = dt

        for i, op in enumerate(self.F_ops):
            op.evaluate(out=self.X_rhs[i])
        RHS = self.M @ self.X.data + dt*self.F.data
        np.copyto(self.X.data, self.LU.solve(RHS))


class CNAB(IMEXTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            # Euler
            LHS = self.M + dt*self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            for i, op in enumerate(self.F_ops):
                op.evaluate(out=self.X_rhs[i])
            self.F_old = np.copy(self.F.data)
            RHS = self.M @ self.X.data + dt*self.F.data
            np.copyto(self.X.data, LU.solve(RHS))
        else:
            if dt != self.dt:
                LHS = self.M + dt/2*self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
                self.dt = dt

            for i, op in enumerate(self.F_ops):
                op.evaluate(out=self.X_rhs[i])
            RHS = self.M @ self.X.data - 0.5*dt*self.L @ self.X.data + 3/2*dt*self.F.data - 1/2*dt*self.F_old
            np.copyto(self.F_old, self.F.data)
            np.copyto(self.X.data, self.LU.solve(RHS))


class BDFExtrapolate(IMEXTimestepper):

    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps =steps
        self.X_array = np.zeros(shape=(self.steps,len(self.X.data)))
        self.F_array = np.zeros(shape=(self.steps,len(self.F.data)))

    def _step(self, dt):
        
        sum_X = np.zeros(len(self.X.data))
        sum_F = np.zeros(len(self.F.data))
        
        if self.iter < self.steps: 
            
            self.X_array[self.iter]=self.X.data
            #print('X.data',self.X.data)
            s=self.iter
           
            j_a = np.arange(s+2)[None,:]
            j_b = np.arange(1,s+2)[None,:]
            k_a = np.arange(s+2)[:,None]
            k_b = np.arange(s+1)[:,None]
            
            if dt != self.dt:                   
                S_a = 1/factorial(k_a)*(-1*j_a*dt)**k_a
                b_a=np.zeros(s+2)      
                b_a[1] = 1
                self.a = np.linalg.solve(S_a, b_a)
                
                S_b = 1/factorial(k_b)*(-j_b)**k_b
                b_b=np.zeros(s+1)      
                b_b[0] = 1
                self.b = np.linalg.solve(S_b, b_b)
                #print('iter',self.iter)
                #print('a',self.a)
                #print('b',self.b)
                #self.dt = dt
                
            LHS = (self.a[0]*self.M) + self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')         
            
            for i, op in enumerate(self.F_ops):
                op.evaluate(out=self.X_rhs[i])
                
            #print('F.data',self.F.data)
                
            self.F_array[self.iter]=self.F.data
                
            for i in range(self.iter+1):
               
                sum_X += self.a[i+1] * self.X_array[self.iter-i]
                sum_F += self.b[i] * self.F_array[self.iter-i]

            RHS = -self.M @ sum_X + sum_F
            #np.copyto(self.F_old, self.F.data)
            np.copyto(self.X.data, self.LU.solve(RHS))
            
            #print('sum_X',sum_F)
            
            #print('F_array',self.F_array)
            #print('X_array',self.X_array)
            
  
        if self.iter >= self.steps:
            #print('______________________________________')
          
            for i in range(self.steps-1):
                self.X_array[i] = self.X_array[i+1]
                self.F_array[i] = self.F_array[i+1]
                
            for i, op in enumerate(self.F_ops):
                op.evaluate(out=self.X_rhs[i])
              
                        
            self.X_array[-1] = self.X.data
            self.F_array[-1] = self.F.data
            
        
            for i in range(self.steps):
               
                sum_X += self.a[i+1] * self.X_array[self.steps-i-1]
                sum_F += self.b[i] * self.F_array[self.steps-i-1]
          
            #print('iter',self.iter)
            
            RHS = -self.M @ sum_X + sum_F
            #np.copyto(self.F_old, self.F.data)
            np.copyto(self.X.data, self.LU.solve(RHS))
            #print('F_data',self.F.data)
            #print('F_array',self.F_array)
            #print('X_array',self.X_array)
            #print('X.data',self.X.data)
            


class Timestepper:

    def __init__(self, u, L_op):
        self.t = 0
        self.iter = 0
        self.u = u
        self.L_op = L_op
        self.dt = None

    def evolve(self, time, dt):
        while self.t < time - 1e-8:
            self.step(dt)

    def step(self, dt):
        self._step(dt)
        self.t += dt
        self.iter += 1


class ForwardEuler(Timestepper):

    def __init__(self, u, F):
        self.t = 0
        self.iter = 0
        self.u = u
        self.RHS = Field(u.grid)
        self.F = F

    def _step(self, dt):
        self.F.evaluate(out=self.RHS)
        self.u.data += dt*self.RHS.data

        
class LaxFriedrichs(Timestepper):

    def __init__(self, u, F):
        self.t = 0
        self.iter = 0
        self.u = u
        self.RHS = Field(u.grid)
        self.F = F
        self.I = Average3(u)

    def _step(self, dt):
        self.F.evaluate(out=self.RHS)
        self.u.data = self.I.matrix @ self.u.data + dt*self.RHS.data


class LeapFrog(Timestepper):

    def __init__(self, u, L_op):
        super().__init__(u, L_op)
        # u_{n-1}
        self.u_old = Field(u.grid, u.data)

    def _step(self, dt):
        if iter == 0:
            I = Identity(self.u.grid, self.L_op.pad) 
            RHS = I + dt*self.L_op
            RHS.operate(self.u, out=self.u)
        else:
            if dt != self.dt:
                self.RHS = 2*dt*self.L_op
                self.dt = dt
            u_temp = self.RHS.operate(self.u)
            u_temp.data += self.u_old.data
            self.u_old.data = self.u.data
            self.u.data = u_temp.data


class LaxWendorff(Timestepper):

    def __init__(self, u, d_op, d2_op):
        self.t = 0
        self.iter = 0
        self.u = u
        self.d_op = d_op
        self.d2_op = d2_op
        self.dt = None
        self.I = Identity(u.grid, d_op.pad)

    def _step(self, dt):
        if dt != self.dt:
            self.RHS = self.I + dt*self.d_op + dt**2/2*self.d2_op
            self.dt = dt
        self.RHS.operate(self.u, out=self.u)


class MacCormack(Timestepper):

    def __init__(self, u, op_f, op_b):
        self.t = 0
        self.iter = 0
        self.u = u
        if op_f.pad != op_b.pad:
            raise ValueError("Forward and Backward operators must have the same padding")
        self.op_f = op_f
        self.op_b = op_b
        self.dt = None
        self.I = Identity(u.grid, op_f.pad)
        self.u1 = Field(u.grid, u.data)

    def _step(self, dt):
        if dt != self.dt:
            self.RHS1 = self.I + dt*self.op_f
            self.RHS2 = 0.5*(self.I + dt*self.op_b)
            self.dt = dt
        self.RHS1.operate(self.u, out=self.u1)
        self.u.data = 0.5*self.u.data + self.RHS2.operate(self.u1).data


class BackwardEuler(Timestepper):

    def __init__(self, u, L_op):
        super().__init__(u, L_op)
        self.I = Identity(u.grid, L_op.pad)

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.I - dt*self.L_op
            self.LU = spla.splu(LHS.matrix.tocsc(), permc_spec='NATURAL')
            self.dt = dt
        self.u.data = self.LU.solve(self.u.data)


class CrankNicolson(Timestepper):

    def __init__(self, u, L_op):
        super().__init__(u, L_op)
        self.I = Identity(u.grid, L_op.pad)

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.I - dt/2*self.L_op
            self.RHS = self.I + dt/2*self.L_op
            self.LU = spla.splu(LHS.matrix.tocsc(), permc_spec='NATURAL')
            self.dt = dt
        self.RHS.operate(self.u, out=self.u)
        self.u.data = self.LU.solve(self.u.data)


