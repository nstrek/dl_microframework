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

    # def __mul__(self, other):
    #     if isinstance(other, Tensor):
    #         other = other.value
    #
    #     return Tensor(self.value * other, requires_grad=self.requires_grad)
    #
    # def __add__(self, other):
    #     if isinstance(other, Tensor):
    #         other = other.value
    #
    #     return Tensor(self.value + other, requires_grad=self.requires_grad)
    #
    # def __radd__(self, other):
    #     return self.__add__(other)
    #
    # def __rmul__(self, other):
    #     return self.__mul__(other)
    #
    # def __sub__(self, other):
    #     return self.__add__(-other)
    #
    # def __rsub__(self, other):
    #     return self.__sub__(other)
    #
    # def __truediv__(self, other):
    #     return Tensor(self.value / other, requires_grad=self.requires_grad)
    #
    # def __neg__(self):
    #     return Tensor(0 - self.value, requires_grad=self.requires_grad)


class Parameters(dict):
    # TODO: Как-то более грамотно стоит здесь выразить, что предполагается, что key экземпляры str,
    #  а item экземляры Tensor, float, int
    valid_key_types = (str, )
    valid_item_types = (Tensor,
                        float, np.float16, np.float32, np.float64,
                        int, np.int16, np.int32, np.int64)

    def __init__(self, *args, **kwargs):
        super(dict, self).__init__(*args, **kwargs)

    def __setitem__(self, key, item):
        if not isinstance(key, self.valid_key_types):
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
        self.dfdx_adjusted_after_backward = None

        self.previous_nodes = []

        self.next_nodes = []

    # TODO: Может быть одновременно вычислять функцию и производную? Сделать режим трейн и инференс
    def compute_function(self, *args: Tuple[np.ndarray]) -> np.ndarray:
        return self.f(*args, params=self.params)

    def compute_derivatives(self, *args: Tuple[np.ndarray]) -> Dict[AnyStr, np.ndarray]:
        return self.df(*args, params=self.params)

    def forward(self, *args: Union[np.ndarray, Tuple[np.ndarray]]) -> np.ndarray:# -> Dict[AnyStr, Union[Tensor, Dict[AnyStr, Tensor]]]:
        # if not isinstance(args[0], Tensor):
        #     raise ValueError(f'*args must be Tensor or Tuple of Tensors {args=}')

        self.last_forward_result = {'f': self.compute_function(*args), 'df': self.compute_derivatives(*args)}
        print(self.name, self.last_forward_result['f'].shape)
        return self.last_forward_result['f']

    def backward(self):
        if len(self.next_nodes) == 0:
            self.dfdx_adjusted_after_backward = self.last_forward_result['df']['dfdx']
        else:
            # print(self.next_nodes[0].last_forward_result['df'], self.dfdx_adjusted_after_backward)
            self.dfdx_adjusted_after_backward = np.dot(self.next_nodes[0].dfdx_adjusted_after_backward, self.next_nodes[0].last_forward_result['df']['dfdx'])

        print(self.dfdx_adjusted_after_backward, 'dfdx_adjusted')


class Loss(Operation):
    def __init__(self,
                 name: AnyStr,
                 f: Callable[[Tensor, Parameters], Tensor],  # np.float64
                 df: Callable[[Tensor, Parameters], Tensor],  # np.float64
                 params: Union[Parameters, None]):
        super(Operation, self).__init__(name=name, f=f, df=df, params=params)


class Activation(Operation):
    def __init__(self,
                 name: AnyStr,
                 f: Callable[[Tensor, Parameters], Tensor],
                 df: Callable[[Tensor, Parameters], Tensor],
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

        f = lambda x, params: \
            (np.matmul(params['A'].value[None, :, :], x[:, :, None]) + params['b'].value[:, None]).squeeze(-1)
        dfdx = lambda x, params: np.matmul(params['A'].value[None, :, :], np.ones_like(x[:, :, None])).squeeze(-1)

        dfdA = lambda x, params: np.tile(np.expand_dims(x, axis=1), (1, params['A'].value.shape[0], 1))
        dfdb = lambda x, params: np.ones((x.shape[0], params['b'].value.shape[0]))

        df = lambda x, params: {'dfdx': dfdx(x, params), 'dfdA': dfdA(x, params), 'dfdb': dfdb(x, params)}

        super(Layer, self).__init__(name='Dense', f=f, df=df, params=dense_params)

    def backward(self):
        super(Layer, self).backward()
        curr_forward = self.last_forward_result

        self.params['A'].grad = curr_forward['df']['dfdA'] * self.dfdx_adjusted_after_backward
        self.params['b'].grad = curr_forward['df']['dfdb'] * self.dfdx_adjusted_after_backward


class Sigmoid(Activation):
    def __init__(self):
        # TODO: Сделать возможность вычислять производную через значения функции
        f = lambda x, params: 1 / (1 + np.exp(-x))
        dfdx = lambda x, params: f(x, params) * (1 - f(x, params))
        df = lambda x, params: {'dfdx': dfdx(x, params)}

        super(Activation, self).__init__(name='Sigmoid', f=f, df=df, params=None)


class BinaryCrossEntropy(Loss):
    def __init__(self):
        # TODO: Хорошо бы где-то сделать проверки shape'ов и это касается не только этого случая
        # TODO: Запилить другие reduction, не только сумму
        f = lambda pred, target, params: -np.sum(np.dot(target, np.log(pred)) + np.dot((1 - target), np.log(1 - pred)), axis=0)
        dfdx = lambda pred, target, params: -np.sum((target / pred) + (1 - target) / (1 - pred), axis=0)
        df = lambda pred, target, params: {'dfdx': dfdx(pred, target, params)}

        bounding_pred = lambda pred: eps + pred * (1 - 2 * eps)

        bounded_f = lambda pred, target, params: f(bounding_pred(pred), target, params)
        bounded_df = lambda pred, target, params: df(bounding_pred(pred), target, params)

        super(Loss, self).__init__(name='CrossEntropy', f=bounded_f, df=bounded_df, params=None)  # TODO: Добавить веса классов


class BatchNormalization(Layer):
    def __init__(self):
        pass


class Model:
    # TODO: Реализовать multiple inputs multiple outputs
    def __init__(self):
        raise NotImplementedError()


class Sequential(Model):
    def __init__(self, network_operations: List[Operation], loss_func: Loss):#, optimizer: Optimizer):
        self.loss_func = loss_func
        self.network_operations = network_operations

        def create_graph():
            self.loss_func.previous_nodes.append(self.network_operations[-1])

            for k in reversed(range(1, len(self.network_operations))):
                self.network_operations[k].previous_nodes.append(self.network_operations[k - 1])

                self.network_operations[k - 1].next_nodes.append(self.network_operations[k])

            self.network_operations[-1].next_nodes.append(self.loss_func)

        create_graph()

    def predict(self, X):
        pred = X
        for operation in self.network_operations:
            # print('check', isinstance(pred, np.ndarray), operation.name)
            pred = operation.forward(pred)

        return pred

    def forward(self, X, Y):
        return self.loss_func.forward(self.predict(X), Y)

    def backward(self):
        self.loss_func.backward()
        for curr_operation in reversed(self.network_operations):
            curr_operation.backward()

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

np.random.seed(0)

model = Sequential([Dense(input_size=10, output_size=1), Sigmoid()], BinaryCrossEntropy())

X = np.random.random((3, 10))
Y = 1 / (1 + np.exp(-np.sum(X, axis=1)))

print(model.predict(X).shape, 'predict')
print(model.forward(X, Y).shape)

print(model.predict(X))
print(model.forward(X, Y))
model.backward()

print(model.network_operations[0].params['A'].grad)