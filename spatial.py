
import numpy as np
import scipy.sparse as sparse
from scipy.special import factorial
import math
from field import LinearOperator, UniformNonPeriodicGrid

def axslice(axis, start, stop, step=None):
    """Slice array along a specified axis."""
    if axis < 0:
        raise ValueError("`axis` must be positive")
    slicelist = [slice(None)] * axis
    slicelist.append(slice(start, stop, step))
    return tuple(slicelist)

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

class FiniteDifferenceUniformGrid(LinearOperator):

    def __init__(self, derivative_order, convergence_order, arg, axis=0, pad=None, stencil_type='centered'):
        if stencil_type == 'centered' and convergence_order % 2 != 0:
            raise ValueError("Centered finite difference has even convergence order")
        if stencil_type == 'forward' or stencil_type == 'backward':
            if derivative_order % 2 == 0:
                raise ValueError("Forward and backward finite difference only for odd derivative order.")

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.pad = pad
        self.axis = axis
        self.grid = arg.domain.grids[axis]
        self._stencil_shape(stencil_type)
        self._make_stencil(self.grid)
        self._build_matrices(self.grid)
        super().__init__(arg)

    def _stencil_shape(self, stencil_type):
        dof = self.derivative_order + self.convergence_order

        if stencil_type == 'centered':
            # cancellation if derivative order is even
            dof = dof - (1 - dof % 2)
            j = np.arange(dof) - dof//2
        if stencil_type == 'forward':
            j = np.arange(dof) - dof//2 + 1
        if stencil_type == 'backward':
            j = np.arange(dof) - dof//2
        if stencil_type == 'full forward':
            j = np.arange(dof)
        if stencil_type == 'full backward':
            j = -np.arange(dof)

        self.dof = dof
        if self.pad == None:
            self.pad = (-np.min(j), np.max(j))
        self.j = j

    def _make_stencil(self, grid):

        # assume constant grid spacing
        self.dx = grid.dx
        i = np.arange(self.dof)[:, None]
        j = self.j[None, :]
        S = 1/factorial(i)*(j*self.dx)**i

        b = np.zeros( self.dof )
        b[self.derivative_order] = 1.

        self.stencil = np.linalg.solve(S, b)

    def _build_matrices(self, grid):
        shape = [grid.N + self.pad[0] + self.pad[1]] * 2
        padded_matrix = sparse.diags(self.stencil, self.j, shape=shape)
        self.padded_matrix = sparse.diags(self.stencil, self.j, shape=shape)
        self.matrix = self._unpadded_matrix(grid)

    def error_estimate(self, lengthscale):
        error_degree = self.dof
        if self.stencil_type == 'centered' and self.derivative_order % 2 == 0:
            error_degree += 1
        error = np.abs(np.sum( self.stencil*(self.j*self.dx/lengthscale)**error_degree ))
        error *= 1/math.factorial(error_degree)
        return error

    def plot_matrix(self):
        self._plot_2D(self.matrix.A)

    def fourier_representation(self):
        kh = np.linspace(-np.pi, np.pi, 100)
        derivative = np.sum(self.stencil[:,None]*np.exp(1j*kh[None,:]*self.j[:,None]),axis=0)*self.dx**self.derivative_order
        return kh, derivative

class BoundaryCondition(LinearOperator):

    def __init__(self, derivative_order, convergence_order, arg, value, axis=0):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.dof = self.derivative_order + self.convergence_order
        self.value = value
        self.axis = axis
        self.grid = arg.domain.grids[axis]
        if not isinstance(self.grid, UniformNonPeriodicGrid):
            raise ValueError("Can only apply BC's on UniformNonPeriodicGrid")
        self._build_vector()
        N = self.grid.N
        self.matrix = self.vector.reshape((1,N))
        super().__init__(arg)

    def _coeffs(self, dx, j):
        i = np.arange(self.dof)[:, None]
        j = j[None, :]
        S = 1/factorial(i)*(j*dx)**i

        b = np.zeros( self.dof )
        b[self.derivative_order] = 1.

        return np.linalg.solve(S, b)
        
    def field_coeff(self, field, axis=None):
        if axis == None:
            axis = self.axis
        if axis != self.axis:
            raise ValueError("Axis must match self.axis")
        if field == self.field:
            return self.matrix
        else:
            return 0*self.matrix
        
        
class Left(BoundaryCondition):

    def _build_vector(self):
        dx = self.grid.dx
        j = 1/2 + np.arange(self.dof)
        
        coeffs = self._coeffs(dx, j)
        
        self.vector = np.zeros(self.grid.N)
        self.vector[:self.dof] = coeffs
        
    def operate(self):
        s = axslice(self.axis, 1, None)
        BC = self.value - apply_matrix(self.matrix[:,1:], self.field.data[s], self.axis)
        BC /= self.matrix[0,0]
        s = axslice(self.axis, 0, 1)
        self.field.data[s] = BC

        
class Right(BoundaryCondition):

    def _build_vector(self):
        dx = self.grid.dx
        j = np.arange(self.dof) - self.dof + 1/2
        
        coeffs = self._coeffs(dx, j)
        
        self.vector = np.zeros(self.grid.N)
        self.vector[-self.dof:] = coeffs
        
    def operate(self):
        s = axslice(self.axis, None, -1)
        BC = self.value - apply_matrix(self.matrix[:,:-1], self.field.data[s], self.axis)
        BC /= self.matrix[0,-1]
        s = axslice(self.axis, -1, None)
        self.field.data[s] = BC
