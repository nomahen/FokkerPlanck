
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from functools import reduce
import numbers

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


class Array:

    def __init__(self, grid, data=None):
        self.grid = grid
        if data is None:
            self.data = np.zeros(grid.N)
        else:
            self.data = data

    def __neg__(self):
        return Array(self.grid, data=-self.data)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return Array(self.grid, data=other*self.data)
        else:
            return other*self

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return Array(self.grid, data=other*self.data)
        else:
            raise NotImplementedError()

class Field:

    def __init__(self, grid, data=None):
        self.grid = grid
        if data is None:
            self.data = np.zeros(grid.N)
        else:
            self.data = np.copy(data)

    def data_padded(self, pad, bc='periodic'):
        start_pad, end_pad = pad
        new_data = np.zeros(start_pad + self.grid.N + end_pad)
        if end_pad == 0:
            s = slice(start_pad, None)
        else:
            s = slice(start_pad, -end_pad)
        new_data[s] = self.data
        if bc == 'periodic':
            if start_pad > 0:
                new_data[:start_pad] = self.data[-start_pad:]
            if end_pad > 0:
                new_data[-end_pad:] = self.data[:end_pad]
        else:
            raise ValueError("Only supports periodic BC")
        return new_data

    def field_coeff(self, field):
        if field == self:
            return Identity(self).matrix
        else:
            return 0*Identity(self).matrix

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
            array = Array(self.grid, self.data*0+other)
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
            return other*Identity(self)
        elif isinstance(other, Array):
            return MultiplyArrayField((other, self))
        else:
            raise ValueError("Can only multiply a Field with another Field, an Operator, a Number, or an Array")

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return other*Identity(self)
        elif isinstance(other, Array):
            return MultiplyArrayField((other, self))
        else:
            raise ValueError("Can only multiply a Field with another Field, an Operator, a Number, or an Array")


class Operator:

    def __init__(self, args):
        self.args = args
        self.grid = self._get_grid()
        for i, arg in enumerate(self.args):
            if isinstance(arg, numbers.Number):
                data = np.zeros(self.grid.N) + arg
                self.args[i] = Array(self.grid, data)
            elif (not isinstance(arg, Operator)) and (not isinstance(arg, Field)) and (not isinstance(arg, Array)):
                raise NotImplementedError("arguments must be either Operators, Fields, Arrays, or Numbers")

    def _get_grid(self):
        for arg in self.args:
            if isinstance(arg, Operator) or isinstance(arg, Field) or isinstance(arg, Array):
                return arg.grid

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
            out = Field(self.grid)
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
        self.grid = self.args[1].grid
        self.out = Field(self.grid)

    def operate(self, out=None):
        if out is None:
            out = Field(self.grid)
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
                for index in arg_indices[1:]:
                    # wrap with Identity here so that we add as LinearOperators
                    arg = Identity(arg) + Identity(args[index])
            new_args.append(arg)
 
        return new_args

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
            out = Field(self.grid)
        args_data = [arg.data for arg in self.args]
        self.data = reduce(np.add, args_data)
        np.copyto(out.data, self.data)
        return out

    def field_coeff(self, field):
        for arg in self.args:
            if isinstance(arg, Field):
                if arg == field:
                    return Identity(arg).matrix
            if isinstance(arg, LinearOperator):
                if arg.field == field:
                    return arg.matrix
        return 0*Identity(arg).matrix


class LinearOperator(Operator):

    def __init__(self, arg):
        self.grid = arg.grid
        if isinstance(arg, Field):
            self.field = arg
        elif isinstance(arg, LinearOperator):
            self.field = arg.field
            self.matrix = self.matrix @ arg.matrix
        self.args = [self.field]

    def operate(self, out=None):
        field = self.field
        self.data = self.matrix @ field.data
        if out is None:
            return Field(field.grid, self.data)
        else:
            np.copyto(out.data, self.data)

    def field_coeff(self, field):
        if field == self.field:
            return self.matrix
        else:
            return 0*Identity(field).matrix

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

    def __add__(self, other):
        if isinstance(other, LinearOperator):
            if self.field == other.field:
                op = LinearOperator(self.field)
                op.matrix = self.matrix + other.matrix
                return op
        if isinstance(other, Field):
            if self.field == other:
                return self + 1*other
        # otherwise Addition
        return Addition((self, other))

    def __neg__(self):
        op = LinearOperator(self.field)
        op.matrix = -self.matrix
        return op

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            op = LinearOperator(self.field)
            op.matrix = other*self.matrix
            return op
        elif isinstance(other, Array):
            return MultiplyArrayField((other, self))
        else:
            return Multiply((self, other))

    def __rmul__(self, other):
        return self*other


class MultiplyArrayField(LinearOperator):

    def __init__(self, args):
        array = args[0]
        self.arg = args[1]
        self.grid = self.arg.grid
        self.matrix = sparse.diags([array.data], [0], shape=[self.grid.N]*2)
        super().__init__(self.arg)


class Identity(LinearOperator):

    def __init__(self, arg):
        self.grid = arg.grid
        self._stencil_shape()
        self._make_stencil()
        self._build_matrices(self.grid)
        super().__init__(arg)

    def _stencil_shape(self):
        self.j = np.arange(1)

    def _make_stencil(self):
        self.stencil = np.array([1])

    def _build_matrices(self, grid):
        shape = [grid.N] * 2
        self.matrix = sparse.diags(self.stencil, self.j, shape=[grid.N]*2)


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
        self.grid = field_list[0].grid
        for field in field_list:
            if self.grid != field.grid:
                raise ValueError("All fields must be on the same grid")

        N = self.grid.N
        data_shape = self.num_fields * N
        self.data = np.zeros(data_shape)

        # copy the data
        for i, field in enumerate(field_list):
            self.data[i*N:(i+1)*N] = np.copy(field.data)
        # update the field data to point to the corresponding view in the System
        for i, field in enumerate(field_list):
            field.data = self.data[i*N:(i+1)*N]
        self.field_list = field_list

