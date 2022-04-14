import numpy as np
from typing import AnyStr, Callable, Tuple, Dict, Union


class Tensor:
    def __init__(self, value: np.array):
        self.value = value
        self.grad = np.empty_like(value)


class Parameters(dict):
    # TODO: Как-то более грамотно надо здесь выразить, что предполагается, что key экземпляры str,
    #  а item экземляры Tensor, float, int
    valid_item_types = (Tensor,
                        float, np.float16, np.float32, np.float64,
                        int, np.int16, np.int32, np.int64)

    def __init__(self, *args, **kwargs):
        super(dict, self).__init__(*args, **kwargs)

    def __setitem__(self, key, item):
        if not isinstance(key, str):
            raise ValueError(f'Parameters expects key is str instance, but your input is {key=} {type(key)=}')

        if not isinstance(item, self.valid_item_types):
            raise ValueError(
                f'Parameters expect item is instance of {self.valid_item_types}, but type of your input is {type(item)=}')

        self.__dict__[key] = item


class Operation:
    def __init__(self,
                 name: AnyStr,
                 f: Callable,
                 df: Callable,
                 params: Parameters):
        self.name = name
        self.f = f
        self.df = df
        self.params = params

    # TODO: Может быть одновременно вычислять функцию и производную? Сделать режим трейн и инференс
    def compute_function(self, *args: Tuple[Tensor]):
        return self.f(*args, params=self.params)

    def compute_gradient(self, *args: Tuple[Tensor]):
        return self.df(*args, params=self.params)

    def forward(self, *args: Tuple[Tensor]):
        return {'f': self.compute_function(*args), 'df': self.compute_gradient(*args)}


class Loss(Operation):
    def __init__(self,
                 name: AnyStr,
                 f: Callable[[Tensor, Tensor], np.float64],
                 df: Callable[[Tensor, Tensor], np.float64],
                 params: Parameters):
        super(Operation, self).__init__(name=name, f=f, df=df, params=params)


class Activation(Operation):
    def __init__(self,
                 name: AnyStr,
                 f: Callable[[Tensor, Tensor], Tensor],
                 df: Callable[[Tensor, Tensor], Tensor],
                 params: Parameters):
        super(Operation, self).__init__(name=name, f=f, df=df, params=params)


class Layer(Operation):
    def __init__(self,
                 name: AnyStr,
                 f: Callable[[Tensor, Tensor], Tensor],
                 df: Callable[[Tensor, Tensor], Tensor],
                 params: Parameters):
        super(Operation, self).__init__(name=name, f=f, df=df, params=params)



class Dense(Layer):
    def __init__(self, input_size, output_size):
        params = Parameters()
        params['A'] = Tensor(np.random.random((input_size, output_size)))
        params['b'] = Tensor(np.random.random(output_size))

        super(Layer, self).__init__(name='Dense',
                                    f=lambda x, params: np.matmul(x, params['A'][:, :, None]).squeeze(-1) + params['b'],
                                    df=lambda x, params: np.ones(),
                                    params=params)


X = np.random.random((6, 10))
A = np.random.random((2, 10))

print(A[None, :, :].shape)

print(np.matmul(A[None, :, :], X[:, :, None]).squeeze(-1))