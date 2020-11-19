
import numpy as np
from field import Field, FieldSystem, Identity, Average3
from scipy.special import factorial
from scipy import sparse
import scipy.sparse.linalg as spla
from collections import deque

# from K. J. Burns
def apply_matrix(matrix, array, axis, **kw):
    """Contract any direction of a multidimensional array with a matrix."""
    dim = len(array.shape)
    # Build Einstein signatures
    mat_sig = [dim, axis]
    arr_sig = list(range(dim))
    out_sig = list(range(dim))
    out_sig[axis] = dim
    # Handle sparse matrices
    if sparse.isspmatrix(matrix):
        matrix = matrix.toarray()
    return np.einsum(matrix, mat_sig, array, arr_sig, out_sig, **kw)

class Timestepper:

    def __init__(self, eq_set):
        self.t = 0
        self.iter = 0
        self.dt = None

        self.X = eq_set.X

    def evolve(self, time, dt):
        while self.t < time - 1e-8:
            self.step(dt)

    def step(self, dt):
        self._step(dt)
        self.t += dt
        self.iter += 1


class ExplicitTimestepper(Timestepper):

    def __init__(self, eq_set):
        super().__init__(eq_set)
        self.F_ops = eq_set.F_ops
        X_rhs = []
        for field in self.X.field_list:
            X_rhs.append(Field(field.domain))
        self.F = FieldSystem(X_rhs)
        self.X_rhs = X_rhs

    def _evaluate_F(self):
        for i, op in enumerate(self.F_ops):
                op.evaluate(out=self.X_rhs[i])


class ForwardEuler(ExplicitTimestepper):

    def _step(self, dt):
        self._evaluate_F()
        self.X.data += dt*self.F.data


class AdamsBashforth(ExplicitTimestepper):

    def __init__(self, steps, eq_set):
        super().__init__(eq_set)
        self.steps = steps
        self.F_list = deque()
        for i in range(self.steps):
            self.F_list.append(np.copy(self.F.data))

    def _step(self, dt):
        self.F_list.rotate()
        self._evaluate_F()
        np.copyto(self.F_list[0], self.F.data)
        if self.iter < self.steps:
            coeffs = self._coeffs(self.iter+1)
        else:
            coeffs = self._coeffs(self.steps)

        for i, coeff in enumerate(coeffs):
            self.X.data += dt*coeff*self.F_list[i].data

    def _coeffs(self, num):

        i = (1 + np.arange(num))[None, :]
        j = (1 + np.arange(num))[:, None]
        S = (-i)**(j-1)/factorial(j-1)

        b = (-1)**(j+1)/factorial(j)

        a = np.linalg.solve(S, b)
        return a

class PredictorCorrector(ExplicitTimestepper):

    def __init__(self, eq_set):
        super().__init__(eq_set)
        self.X_old = np.copy(self.X.data)
        self.F_old = np.copy(self.F.data)

    def _step(self, dt):
        # predictor
        self._evaluate_F()
        np.copyto(self.X_old, self.X.data)
        self.X.data += dt*self.F.data
        # corrector
        np.copyto(self.F_old, self.F.data)
        self._evaluate_F()
        np.copyto(self.X.data, self.X_old + dt/2*(self.F_old + self.F.data))


class ImplicitTimestepper(Timestepper):

    def __init__(self, eq_set, axis):
        super().__init__(eq_set)
        self.M = eq_set.M
        self.L = eq_set.L
        self.axis = axis
        self.domain = eq_set.domain
        self.data = np.zeros(self.X.data.shape)

    def _transpose_pre(self):
        # transpose
        order1 = np.array([0])
        order2 = np.array([self.axis+1])
        order3 = 1+np.arange(self.axis)
        order4 = 1+np.arange(self.axis+1, self.domain.dimension)
        order = np.concatenate((order1, order2, order3, order4))
        np.copyto(self.data, np.transpose(self.X.data, order))
        # make view
        shape = list(self.data.shape[1:])
        shape[0] *= self.data.shape[0]
        self.data = self.data.reshape(shape)

    def _transpose_post(self):
        # make view
        Xshape = self.X.data.shape
        shape = list(self.data.shape)
        shape[0] = Xshape[self.axis+1]
        shape.insert(0, Xshape[0])
        self.data = self.data.reshape(shape)
        # transpose
        order1 = np.array([0])
        order2 = 2+np.arange(self.axis)
        order3 = np.array([1])
        order4 = 1+np.arange(self.axis+1, self.domain.dimension)
        order = np.concatenate((order1, order2, order3, order4))
        np.copyto(self.X.data, np.transpose(self.data, order))


class BackwardEuler(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt*self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.dt = dt
        self._transpose_pre()
        self.data = self.LU.solve(self.data)
        self._transpose_post()


class CrankNicolson(ImplicitTimestepper):

    def __init__(self, eq_set, axis):
        super().__init__(eq_set, axis)
        self.RHS = np.copy(self.data)

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt/2*self.L
            self.RHS_matrix = self.M - dt/2*self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.dt = dt
        
        if (self.axis == 'full'):
            np.copyto(self.data, self.X.data)
            data_shape = self.data.shape
            flattened_data = self.data.reshape(np.prod(data_shape))
            self.RHS = self.RHS.reshape(flattened_data.shape)
            apply_matrix(self.RHS_matrix, flattened_data, 0, out=self.RHS)
            self.data = self.LU.solve(self.RHS).reshape(data_shape)
            np.copyto(self.X.data, self.data)
            
        else:
            self._transpose_pre()
            # change view
            self.RHS = self.RHS.reshape(self.data.shape)
            apply_matrix(self.RHS_matrix, self.data, 0, out=self.RHS)
            self.data = self.LU.solve(self.RHS)
            self._transpose_post()
            
from scipy.sparse.linalg import lgmres            
class CrankNicolson2(ImplicitTimestepper):

    def __init__(self, eq_set, axis):
        super().__init__(eq_set, axis)
        self.RHS = np.copy(self.data)

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.M + dt/2*self.L
            self.RHS_matrix = self.M - dt/2*self.L
            #self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.dt = dt
        
        if (self.axis == 'full'):
            np.copyto(self.data, self.X.data)
            data_shape = self.data.shape
            flattened_data = self.data.reshape(np.prod(data_shape))
            self.RHS = self.RHS.reshape(flattened_data.shape)
            apply_matrix(self.RHS_matrix, flattened_data, 0, out=self.RHS)
            #self.data = self.LU.solve(self.RHS).reshape(data_shape)
            self.data,exitCode = lgmres(self.LHS,self.RHS)
            self.data=self.data.reshape(data_shape)
            np.copyto(self.X.data, self.data)
            
        else:
            self._transpose_pre()
            # change view
            self.RHS = self.RHS.reshape(self.data.shape)
            apply_matrix(self.RHS_matrix, self.data, 0, out=self.RHS)
            self.data = self.LU.solve(self.RHS)
            self._transpose_post()
            
            
from scipy.sparse.linalg import gmres            
class CrankNicolson3(ImplicitTimestepper):

    def __init__(self, eq_set, axis):
        super().__init__(eq_set, axis)
        self.RHS = np.copy(self.data)

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.M + dt/2*self.L
            self.RHS_matrix = self.M - dt/2*self.L
            #self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.dt = dt
        
        if (self.axis == 'full'):
            np.copyto(self.data, self.X.data)
            data_shape = self.data.shape
            flattened_data = self.data.reshape(np.prod(data_shape))
            self.RHS = self.RHS.reshape(flattened_data.shape)
            apply_matrix(self.RHS_matrix, flattened_data, 0, out=self.RHS)
            #self.data = self.LU.solve(self.RHS).reshape(data_shape)
            self.data,exitCode = gmres(self.LHS,self.RHS)
            self.data=self.data.reshape(data_shape)
            np.copyto(self.X.data, self.data)
            
        else:
            self._transpose_pre()
            # change view
            self.RHS = self.RHS.reshape(self.data.shape)
            apply_matrix(self.RHS_matrix, self.data, 0, out=self.RHS)
            self.data = self.LU.solve(self.RHS)
            self._transpose_post()


class IMEXTimestepper(ExplicitTimestepper):

    def __init__(self, eq_set):
        super().__init__(eq_set)
        self.M = eq_set.M
        self.L = eq_set.L


class Euler(IMEXTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt*self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.dt = dt

        self._evaluate_F()
        RHS = self.M @ self.X.data + dt*self.F.data
        np.copyto(self.X.data, self.LU.solve(RHS))


class CNAB(IMEXTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            # Euler
            LHS = self.M + dt*self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self._evaluate_F()
            self.F_old = np.copy(self.F.data)
            RHS = self.M @ self.X.data + dt*self.F.data
            np.copyto(self.X.data, LU.solve(RHS))
        else:
            if dt != self.dt:
                LHS = self.M + dt/2*self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
                self.dt = dt

            self._evaluate_F()
            RHS = self.M @ self.X.data - 0.5*dt*self.L @ self.X.data + 3/2*dt*self.F.data - 1/2*dt*self.F_old
            np.copyto(self.F_old, self.F.data)
            np.copyto(self.X.data, self.LU.solve(RHS))


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


