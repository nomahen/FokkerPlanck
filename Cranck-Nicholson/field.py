
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from functools import reduce
import numbers

# next two functions from K. J. Burns
def reshape_vector(data, dim=2, axis=-1):
    """Reshape 1-dim array as a multidimensional vector."""
    # Build multidimensional shape
    shape = [1] * dim
    shape[axis] = data.size
    return data.reshape(shape)

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

class UniformPeriodicGrid:

    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N


class PeriodicGrid:

    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)


class Domain:

    def __init__(self, grids):
        self.dimension = len(grids)
        self.grids = grids
        shape = []
        for grid in self.grids:
            shape.append(grid.N)
        self.shape = shape

    def values(self):
        v = []
        for i, grid in enumerate(self.grids):
            grid_v = grid.values
            grid_v = reshape_vector(grid_v, self.dimension, i)
            v.append(grid_v)
        return v

    def plotting_arrays(self):
        v = []
        expanded_shape = np.array(self.shape, dtype=np.int)
        expanded_shape += 1
        for i, grid in enumerate(self.grids):
            grid_v = grid.values
            grid_v = np.concatenate((grid_v, [grid.length]))
            grid_v = reshape_vector(grid_v, self.dimension, i)
            grid_v = np.broadcast_to(grid_v, expanded_shape)
            v.append(grid_v)
        return v


class Array:

    def __init__(self, domain, data=None):
        self.domain = domain
        if data is None:
            self.data = np.zeros(self.domain.shape)
        else:
            self.data = data

    def __neg__(self):
        return Array(self.domain, data=-self.data)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return Array(self.domain, data=other*self.data)
        else:
            return other*self

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return Array(self.domain, data=other*self.data)
        else:
            raise NotImplementedError()


class Field:

    def __init__(self, domain, data=None):
        self.domain = domain
        if data is None:
            self.data = np.zeros(self.domain.shape)
        else:
            self.data = np.copy(data)

    def field_coeff(self, field, axis='full'):
        if field == self:
            return Identity(self).get_matrix(axis)
        else:
            return Const(self, 0).get_matrix(axis)

    def flatten_data(self):
        """ return flatten view of data """
        return self.data.reshape(np.prod(self.domain.shape))

    def multi_data(self):
        """ return multi-dimensional view of data """
        return self.data.reshape(self.domain.shape)

    def __add__(self, other):
        if isinstance(other, Field):
            if other == self:
                return 2*self
            else:
                return Addition((self, other))
        elif isinstance(other, LinearOperator):
            if other.field == self:
                return other + 1*self
            else:
                return Addition((self, other))
        elif isinstance(other, numbers.Number):
            # cast up to Array
            array = Array(self.domain, self.data*0+other)
            return Addition((self, array))
        elif isinstance(other, Operator):
            return Addition((self, other))
        else:
            raise NotImplementedError("Adding together these things is not working yet.")

    def __radd__(self, other):
        if isinstance(other, numbers.Number):
            return self+other

    def __neg__(self):
        return (-1)*self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, Field) or isinstance(other, Operator):
            return Multiply((self, other))
        elif isinstance(other, numbers.Number):
            return Const(self, other)
        elif isinstance(other, Array):
            return MultiplyArrayField((other, self))
        else:
            raise ValueError("Can only multiply a Field with another Field, an Operator, a Number, or an Array")

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return Const(self, other)
        elif isinstance(other, Array):
            return MultiplyArrayField((other, self))
        else:
            raise ValueError("Can only multiply a Field with another Field, an Operator, a Number, or an Array")


class Operator:

    def __init__(self, args):
        self.args = args
        self.domain = self._get_domain()
        for i, arg in enumerate(self.args):
            if isinstance(arg, numbers.Number):
                data = np.zeros(self.domain.shape) + arg
                self.args[i] = Array(self.domain, data)
            elif (not isinstance(arg, Operator)) and (not isinstance(arg, Field)) and (not isinstance(arg, Array)):
                raise NotImplementedError("arguments must be either Operators, Fields, Arrays, or Numbers")

    def _get_domain(self):
        for arg in self.args:
            if isinstance(arg, Operator) or isinstance(arg, Field) or isinstance(arg, Array):
                return arg.domain

    def evaluate(self, out=None):
        for arg in self.args:
            if isinstance(arg, Operator):
                arg.evaluate()
            # otherwise a Field or Array
        if out is None:
            return self.operate()
        else:
            self.operate(out=out)

    def __add__(self, other):
        return Addition((self, other))

    def __radd__(self, other):
        return Addition((self, other))

    def __neg__(self):
        return (-1)*self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return MultiplyNumberField((other, self))
        elif isinstance(other, Operator) or isinstance(other, Array) or isinstance(other, Field):
            return Multiply((self, other))

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return MultiplyNumberField((other, self))


class Multiply(Operator):

    def __init__(self, args):
        super().__init__(args)
        if len(self.args) != 2:
            raise ValueError("Can only multiply two things together")

    def operate(self, out=None):
        if out is None:
            out = Field(self.domain)
        self.data = self.args[0].data * self.args[1].data
        np.copyto(out.data, self.data)
        return out


class MultiplyNumberField(Operator):

    def __init__(self, args):
        self.args = args
        if (not isinstance(args[0], numbers.Number)):
            raise ValueError("First argument must be a Number")
        if (not isinstance(args[1], Operator)) and (not isinstance(args[1], Field)):
            raise ValueError("Second argument must be an Operator or Field")
        self.domain = self.args[1].domain
        self.out = Field(self.domain)

    def operate(self, out=None):
        if out is None:
            out = Field(self.domain)
        self.data = self.args[0] * self.args[1].data
        np.copyto(out.data, self.data)
        return out


class Addition(Operator):

    def __init__(self, args):
        if len(args) < 2:
            raise ValueError("Must add at least two arguments")

        expanded_args = []
        for arg in args:
            if isinstance(arg, Addition):
                for arg_arg in arg.args:
                    expanded_args.append(arg_arg)
            else:
                expanded_args.append(arg)
        args = self._reduce_args(expanded_args)
        super().__init__(args)

    def _reduce_args(self, args):
        # sum linear operators which have the same field

        # make new list of arguments
        new_args = []

        arg_list = {}
        for i, arg in enumerate(args):
            # check to see if Field or LinearOperator
            field = self._get_field(arg)
            if field:
                # keep track of Operand by its Field
                if field in arg_list.keys():
                    arg_list[field].append(i)
                else:
                    arg_list[field] = [i]
            else:
                # otherwise just add it onto the list
                new_args.append(arg)

        # sum together Fields and LinearOperators which have the same Field
        for field in arg_list:
            arg_indices = arg_list[field]
            num_args = len(arg_indices)
            arg = args[arg_indices[0]]
            if num_args > 1:
                # wrap with Identity here so that we add as LinearOperators
                arg_op1 = self._op(arg)
                for index in arg_indices[1:]:
                    # wrap with Identity here so that we add as LinearOperators
                    arg_op2 = self._op(args[index])
                    arg = arg_op1 + arg_op2
            new_args.append(arg)
 
        return new_args

    @staticmethod
    def _op(arg):
        if isinstance(arg, Field):
            return Identity(arg)
        else:
            return arg

    @staticmethod
    def _get_field(arg):
        if isinstance(arg, Field):
            return arg
        elif isinstance(arg, LinearOperator):
            return arg.field
        else:
            return None

    def operate(self, out=None):
        if out is None:
            out = Field(self.domain)
        args_data = [arg.data for arg in self.args]
        self.data = reduce(np.add, args_data)
        np.copyto(out.data, self.data)
        return out

    def field_coeff(self, field, axis='full'):
        for arg in self.args:
            if isinstance(arg, Field):
                if arg == field:
                    return Identity(arg).get_matrix(axis)
            if isinstance(arg, LinearOperator):
                if arg.field == field:
                    return arg.get_matrix(axis)
            #if isinstance(arg, Addition):
            #    coeff = arg.field_coeff(field)
            #    if not (coeff is None):
            #        return coeff
        return Const(field, 0).get_matrix(axis)


class LinearOperator(Operator):

    def __init__(self, arg):
        self.domain = arg.domain
        if isinstance(arg, Field):
            self.field = arg
        elif isinstance(arg, LinearOperator):
            self.field = arg.field
            if self.axis == arg.axis:
                self.matrix = self.matrix @ arg.matrix
            else:
                self.axis = 'full'
                self.matrix = self._full_matrix @ arg._full_matrix
        self.args = [self.field]

    def get_matrix(self, axis):
        if axis == self.axis:
            return self.matrix
        elif axis == 'full':
            return self._full_matrix
        else:
            raise ValueError('Cannot get matrix with different axis')

    def operate(self, out=None):
        field = self.field
        if self.axis == 'full':
            self.data = self.matrix @ field.flatten_data()
            self.data = self.data.reshape(field.domain.shape)
        else:
            self.data = apply_matrix(self.matrix, field.multi_data(), self.axis)
        if out is None:
            return Field(field.domain, self.data)
        else:
            np.copyto(out.data, self.data)

    def field_coeff(self, field, axis='full'):
        if field == self.field:
            return self.get_matrix(axis)
        else:
            return Const(field, 0).get_matrix(axis)

    @property
    def _full_matrix(self):
        """ Kroneckers matrix to full dimensionality """
        if self.axis == 'full':
            return self.matrix
        matrix = sparse.identity(1)
        for i, grid in enumerate(self.domain.grids):
            if self.axis == i:
                new_matrix = self.matrix
            else:
                new_matrix = sparse.identity(grid.N)
            matrix = sparse.kron(matrix, new_matrix)
        return matrix

    def _unpadded_matrix(self, grid):
        matrix = sparse.diags(self.stencil, self.j, shape=[grid.N]*2)
        matrix = matrix.tocsr()
        jmin = -np.min(self.j)
        if jmin > 0:
            for i in range(jmin):
                matrix[i,-jmin+i:] = self.stencil[:jmin-i]

        jmax = np.max(self.j)
        if jmax > 0:
            for i in range(jmax):
                matrix[-jmax+i,:i+1] = self.stencil[-i-1:]
        return matrix

    @staticmethod
    def _plot_2D(matrix, title='FD matrix', output='matrix.pdf'):
        lim_margin = -0.05
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot()
        I, J = matrix.shape
        matrix_mag = np.log10(np.abs(matrix))
        ax.pcolor(matrix_mag[::-1])
        ax.set_xlim(-lim_margin, I+lim_margin)
        ax.set_ylim(-lim_margin, J+lim_margin)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', 'box')
        plt.title(title)
        plt.tight_layout()
#        plt.savefig(output)
#        plt.clf()

    def __add__(self, other):
        if isinstance(other, LinearOperator):
            if self.field == other.field:
                op = LinearOperator(self.field)
                if isinstance(other, Const):
                    op.axis = self.axis
                    op.matrix = self.matrix + other.get_matrix(self.axis)
                    return op
                # if not Const
                if self.axis == other.axis:
                    op.axis = self.axis
                    op.matrix = self.matrix + other.matrix
                else:
                    op.axis = 'full'
                    op.matrix = self._full_matrix + other._full_matrix
                return op
        if isinstance(other, Field):
            if self.field == other:
                return self + 1*other
        # otherwise Addition
        return Addition((self, other))

    def __neg__(self):
        op = LinearOperator(self.field)
        op.matrix = -self.matrix
        op.axis = self.axis
        return op

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            op = LinearOperator(self.field)
            op.axis = self.axis
            op.matrix = other*self.matrix
            return op
        elif isinstance(other, Array):
            return MultiplyArrayField((other, self))
        else:
            return Multiply((self, other))
#            raise ValueError("Can only multiply linear operators by numbers")

    def __rmul__(self, other):
        return self*other


class MultiplyArrayField(LinearOperator):

    def __init__(self, args):
        array = args[0]
        self.arg = args[1]
        self.domain = self.arg.grid
        self.matrix = sparse.diags([array.data], [0], shape=[self.grid.N]*2)
        super().__init__(self.arg)


class Const(LinearOperator):

    def __init__(self, field, const):
        self.field = field
        self.domain = field.domain
        self.const = const
        self.args = [field]

    def get_matrix(self, axis):
        if axis == 'full':
            size = np.prod(self.domain.shape)
        else:
            size = self.domain.shape[axis]
        return self.const*sparse.identity(size)

    def operate(self, out=None):
        self.data = self.const*self.field.data
        if out is None:
            return Field(self.domain, self.data)
        else:
            np.copyto(out.data, self.data)

    def __add__(self, other):
        if isinstance(other, Const):
            if self.field == other.field:
                return Const(self.field, self.const + other.const)
        elif isinstance(other, LinearOperator):
            return other + self
        return Addition((self, other))

    def __neg__(self):
        return Const(self.field, -self.const)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return Const(self.field, other*self.const)
        elif isinstance(other, Array):
            raise NotImplementedError()
            #return MultiplyArrayField((other, self))
        else:
            return Multiply((self, other))


class Identity(Const):

    def __init__(self, field):
        super().__init__(field, 1)


class Average3(LinearOperator):

    def __init__(self, arg):
        self.pad = (1, 1)
        self.grid = arg.grid
        self._stencil_shape()
        self._make_stencil()
        self._build_matrices(self.grid)
        super().__init__(arg)

    def _stencil_shape(self):
        self.j = np.arange(3) - 1

    def _make_stencil(self):
        self.stencil = np.array([1/2, 0, 1/2])

    def _build_matrices(self, grid):
        shape = [grid.N + self.pad[0] + self.pad[1]] * 2
        self.padded_matrix = sparse.diags(self.stencil, self.j, shape=shape)
        self.matrix = self._unpadded_matrix(grid)


class FieldSystem:

    def __init__(self, field_list):
        self.num_fields = len(field_list)
        self.domain = field_list[0].domain
        for field in field_list:
            if self.domain != field.domain:
                raise ValueError("All fields must be on the same grid")

        shape = self.domain.shape
        data_shape = [self.num_fields,] + list(shape)
        self.data = np.zeros(data_shape)

        # copy the data
        for i, field in enumerate(field_list):
            self.data[i] = np.copy(field.data)
        # update the field data to point to the corresponding view in the System
        for i, field in enumerate(field_list):
            field.data = self.data[i]
        self.field_list = field_list



