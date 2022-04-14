import numpy as np
from typing import AnyStr, Callable, Tuple, Dict, Union, List

eps = 1e-16


class Tensor:
    def __init__(self, value: np.array, requires_grad=False):
        self.value = value

        self.requires_grad = requires_grad
        if self.requires_grad:
            self.grad = np.empty_like(value)
        else:
            self.grad = None


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

        super(Parameters, self).__setitem__(key, item)


class Operation:
    def __init__(self,
                 name: AnyStr,
                 f: Callable,
                 df: Callable,
                 params: Union[Parameters, None]):
        self.name = name
        self.f = f
        self.df = df
        self.params = params

        self.last_forward_result = None

        self.previous_nodes = []

    # TODO: Может быть одновременно вычислять функцию и производную? Сделать режим трейн и инференс
    def compute_function(self, *args: Tuple[Tensor]):
        return self.f(*args, params=self.params)

    def compute_gradient(self, *args: Tuple[Tensor]):
        return self.df(*args, params=self.params)

    def forward(self, *args: Union[Tensor, Tuple[Tensor]]):
        self.last_forward_result = {'f': self.compute_function(*args), 'df': self.compute_gradient(*args)}
        return self.last_forward_result


class Loss(Operation):
    def __init__(self,
                 name: AnyStr,
                 f: Callable[[Tensor, Tensor], np.float64],
                 df: Callable[[Tensor, Tensor], np.float64],
                 params: Union[Parameters, None]):
        super(Operation, self).__init__(name=name, f=f, df=df, params=params)


class Activation(Operation):
    def __init__(self,
                 name: AnyStr,
                 f: Callable[[Tensor, Tensor], Tensor],
                 df: Callable[[Tensor, Tensor], Tensor],
                 params: Union[Parameters, None]):
        super(Operation, self).__init__(name=name, f=f, df=df, params=params)


class Layer(Operation):
    def __init__(self,
                 name: AnyStr,
                 f: Callable[[Tensor, Tensor], Tensor],
                 df: Callable[[Tensor, Tensor], Tensor],
                 params: Parameters):
        super(Operation, self).__init__(name=name, f=f, df=df, params=params)


class Optimizer:
    pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        dense_params = Parameters()
        dense_params['A'] = Tensor(np.random.random((output_size, input_size)))
        dense_params['b'] = Tensor(np.random.random(output_size))

        f = lambda x, params: (np.matmul(params['A'].value[None, :, :], x.value[:, :, None]) + params['b'].value[:, None]).squeeze(-1)
        df = lambda x, params: np.tile(np.expand_dims(x.value, axis=1), (1, params['A'].value.shape[0], 1))

        super(Layer, self).__init__(name='Dense', f=f, df=df, params=dense_params)


class Sigmoid(Activation):
    def __init__(self):
        # TODO: Сделать возможность вычислять производную через значения функции
        f = lambda x, params: 1 / (1 + np.exp(-x))
        df = lambda x, params: f(x, params) * (1 - f(x, params))

        super(Activation, self).__init__(name='Sigmoid', f=f, df=df, params=None)


class BinaryCrossEntropy(Loss):
    def __init__(self):
        # TODO: Хорошо бы где-то сделать проверки shape'ов и это касается не только этого случая
        f = lambda pred, target, params: target * np.log(pred) + (1 - target) * np.log(1 - pred)
        df = lambda pred, target, params: (target / pred) + (1 - target) / (1 - pred)

        super(Loss, self).__init__(name='CrossEntropy', f=f, df=df, params=None)  # TODO: Добавить веса классов


class BatchNormalization(Layer):
    def __init__(self):
        pass


class Model:
    # TODO: Реализовать multiple inputs multiple outputs
    def __init__(self):
        raise NotImplementedError()


class Sequential(Model):
    def __init__(self, network_operations: List[Operation], loss_func: Loss, optimizer: Optimizer):
        self.loss_func = loss_func
        self.network_operations = network_operations

        def create_graph():
            self.loss_func.previous_nodes.append(self.network_operations[-1])

            for k in reversed(range(1, len(self.network_operations))):
                self.network_operations[k].previous_nodes.append(self.network_operations[k - 1])

        create_graph()


    def predict(self, X):
        pred = X
        for operation in self.network_operations:
            pred = operation.forward(pred)

        return pred

    def forward(self, X, Y):
        return self.loss_func.forward(self.predict(X), Y)

    def backward(self):
        df_prev = self.loss_func.last_forward_result['df']

        for operation in self.network_operations[::-1]:
            if isinstance(operation, Layer):
                for key, param in operation.params.items():
                    # TODO: Пока считаем, что градиент нужен только по тензорам из Parameters.
                    #  По инпуту не взять без обертывания
                    if not isinstance(param, Tensor) and param.requires_grad:
                        continue

                    df_curr = df_prev
                    output_curr = operation.last_forward_result['f']

                    pass

                    # activation_output_prev =
            elif isinstance(operation, Activation):
                pass




# dense = Dense(10, 2)
#
# X = Tensor(np.random.random((3, 10)))
#
# pred = dense.forward(X)['df']
#
# print(pred.shape)
# print(pred)
# print(np.sum(pred, axis=0))
#
# import torch
#
# torch_A = torch.from_numpy(dense.params['A'].value)
# torch_A.requires_grad_()
# torch_X = torch.from_numpy(np.expand_dims(X.value, axis=2))
#
# f = torch.sum(torch.matmul(torch_A, torch_X))
# print(f.shape)
# f.backward()
#
# print(torch_A.grad.numpy())
