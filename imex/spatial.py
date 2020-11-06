
import numpy as np
import scipy.sparse as sparse
from scipy.special import factorial
import math
from field import LinearOperator


class FiniteDifferenceUniformGrid(LinearOperator):

    def __init__(self, derivative_order, convergence_order, arg, pad=None, stencil_type='centered'):
        if stencil_type == 'centered' and convergence_order % 2 != 0:
            raise ValueError("Centered finite difference has even convergence order")
        if stencil_type == 'forward' or stencil_type == 'backward':
            if derivative_order % 2 == 0:
                raise ValueError("Forward and backward finite difference only for odd derivative order.")

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.pad = pad
        self.grid = arg.grid
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


