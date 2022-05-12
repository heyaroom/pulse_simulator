import numpy as np

class TensorProduct:
    def __init__(self, *dims):
        self.dims = np.array(dims)
        self.size = len(dims)
        self.total_dim = np.prod(self.dims)
        self.operator = np.identity(self.total_dim, dtype=np.complex128).reshape(dims*2)

    def prod(self, operator, target):
        if type(target) is int:
            target = [target]
        dims = self.dims[list(target)]
        operator = np.asarray(operator, dtype=np.complex128).reshape(list(dims)*2)

        c_idx = list(range(2 * self.size))
        t_idx = list(range(2 * self.size))
        for i, _t in enumerate(target):
            t_idx[_t] = 2 * self.size + i
        o_idx = list(range(2 * self.size, 2 * self.size + len(target))) + list(target)

        character = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        o_index = ''.join([character[i] for i in o_idx])
        c_index = ''.join([character[i] for i in c_idx])
        t_index = ''.join([character[i] for i in t_idx])
        subscripts = '{},{}->{}'.format(o_index, c_index, t_index)
        self.operator = np.einsum(subscripts, operator, self.operator)

    def get_operator(self):
        return self.operator.reshape([self.total_dim, self.total_dim])