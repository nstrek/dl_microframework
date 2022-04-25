import sys
import inspect
from framework import *
from typing import List, Callable
import torch
import numpy as np

np.random.seed(0)
TEST_EPS = 1e-12

torch_bounding_pred = lambda pred: eps + pred * (1 - 2 * eps)
torch_bce = lambda pred, target: -torch.sum(target[:, 0] * torch.log(torch_bounding_pred(pred[:, 0])) + (1 - target[:, 0]) * torch.log(1 - torch_bounding_pred(pred[:, 0])))
torch_sigmoid = lambda x: 1 / (1 + torch.exp(-x))


def try_except_print(func):
    def new_func():
        try:
            return func()
        except Exception as er:
            print(er)
            return False

    new_func.__name__ = func.__name__
    return new_func


# TODO: Сделать параметризованный декоратор?
def repeat(func):
    n = 100

    def new_func():
        res = True
        for _ in range(n):
            res = res and func()
        return res

    new_func.__name__ = func.__name__
    return new_func


@repeat
@try_except_print
def test_parameters_0():
    p = Parameters()

    p['5'] = Tensor(np.zeros(1))

    try:
        p[5] = Tensor(np.ones(1))
        return False
    except ValueError:
        pass

    return True


# TODO: Обновить тест
# def test_dense_derivative_0():
#     try:
#         dense = Dense(10, 2)
#
#         X = Tensor(np.random.random((3, 10)))
#
#         pred = dense.forward(X)
#
#         torch_A = torch.from_numpy(dense.params['A'].value)
#         torch_A.requires_grad_()
#         torch_X = torch.from_numpy(np.expand_dims(X.value, axis=2))
#
#         f = torch.sum(torch.matmul(torch_A, torch_X))
#         f.backward()
#
#         print(torch_A.grad.numpy())
#         print(pred['df']['dfdA'])
#
#         if np.linalg.norm(torch_A.grad.numpy() - pred['df']['dfdA']) > TEST_EPS:
#             return False
#     except Exception as er:
#         print(er)
#         return False
#
#     return True

@repeat
@try_except_print
def test_bce():
    bce_model = Sequential([Input(1, 1)], BinaryCrossEntropy())
    torch_bce_model = lambda x, y: torch_bce(x, y)

    X = np.random.random((5, 1))
    Y = np.array(X > np.mean(X), dtype=np.int64)

    res = bce_model.forward(X, Y)
    bce_model.backward()

    torch_X = torch.from_numpy(X)
    torch_X.requires_grad_()

    torch_Y = torch.from_numpy(Y)

    torch_res = torch_bce_model(torch_X, torch_Y)
    torch_res.backward()

    loss_error = np.sum(res) - torch_res.item()
    if loss_error > TEST_EPS or np.isnan(loss_error):
        return False

    grad_error = np.linalg.norm(torch_X.grad.detach().numpy() - bce_model.loss_func.last_forward_result['df']['dfdx'])

    if grad_error > TEST_EPS or np.isnan(grad_error):
        return False

    return True


@repeat
@try_except_print
def test_sigmoid_0():
    sigmoid_model = Sequential([Input(1, 1), Sigmoid()], BinaryCrossEntropy())
    torch_sigmoid_model = lambda x, y: torch_bce(torch_sigmoid(x), y)

    X = np.random.random((5, 1))
    Y = 10 / (1 + np.exp(-np.sum(X, axis=1, keepdims=True))) + 0.5

    res = sigmoid_model.forward(X, Y)
    sigmoid_model.backward()

    torch_X = torch.from_numpy(X)
    torch_X.requires_grad_()

    torch_Y = torch.from_numpy(Y)

    torch_res = torch_sigmoid_model(torch_X, torch_Y)
    torch_res.backward()

    loss_error = np.sum(res) - torch_res.item()

    if loss_error > TEST_EPS or np.isnan(loss_error):
        print(f'sigmoid {loss_error=}')
        return False

    grad_error = np.linalg.norm(
        torch_X.grad.detach().numpy() - sigmoid_model.network_operations[1].last_forward_result['df']['dfdx'])

    if grad_error > TEST_EPS or np.isnan(grad_error):
        print(f'sigmoid {grad_error=}')
        return False

    return True


@repeat
@try_except_print
def test_sequential_0():
    model = Sequential([Dense(input_size=10, output_size=5), Sigmoid(), Dense(input_size=5, output_size=1), Sigmoid()],
                       BinaryCrossEntropy())

    X = np.random.random((4, 10))
    Y = 1 / (1 + np.exp(-np.sum(X, axis=1, keepdims=True)))

    model.forward(X, Y)
    model.backward()

    torch_X = torch.from_numpy(X).cpu()
    torch_Y = torch.from_numpy(Y).cpu()

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.layer1 = torch.nn.Linear(10, 5)
            self.layer2 = torch.nn.Linear(5, 1)
            self.sigmoid = lambda x: 1 / (1 + torch.exp(-x))  # torch.nn.Sigmoid()

        def forward(self, x):
            return self.sigmoid(self.layer2(self.sigmoid(self.layer1(x))))

    torch_model = Network().cpu()
    with torch.no_grad():
        torch_model.layer1.weight = torch.nn.Parameter(torch.from_numpy(model.network_operations[0].params['A'].value))
        torch_model.layer1.bias = torch.nn.Parameter(torch.from_numpy(model.network_operations[0].params[
                                                                          'b'].value * 1))  # Если не умножать на ноль, то разницы нет, а если умножать, то появляется
        torch_model.layer2.weight = torch.nn.Parameter(torch.from_numpy(model.network_operations[2].params['A'].value))
        torch_model.layer2.bias = torch.nn.Parameter(torch.from_numpy(model.network_operations[2].params[
                                                                          'b'].value * 1))  # Если не умножать на ноль, то разницы нет, а если умножать, то появляется

    pred = model.predict(X)
    torch_pred = torch_model.forward(torch_X)
    torch_loss_func = torch_bce  # torch.nn.CrossEntropyLoss(reduction='sum')
    torch_loss = torch_loss_func(torch_pred, torch_Y)
    torch_loss.backward()

    pred_error = np.linalg.norm(pred - torch_pred.detach().numpy())

    if pred_error > TEST_EPS or np.isnan(pred_error):
        print(f'{pred_error=}')
        return False

    loss_error = np.abs(np.sum(model.forward(X, Y), axis=0).item() - torch_loss.item())

    if loss_error > TEST_EPS or np.isnan(loss_error):
        print(f'{loss_error=}')
        return False

    grad_error = np.linalg.norm(
        model.network_operations[2].params['A'].grad - torch_model.layer2.weight.grad.detach().numpy())
    grad_error += np.linalg.norm(
        model.network_operations[0].params['A'].grad - torch_model.layer1.weight.grad.detach().numpy())
    grad_error += np.linalg.norm(
        model.network_operations[0].params['b'].grad - torch_model.layer1.bias.grad.detach().numpy())
    grad_error += np.linalg.norm(
        model.network_operations[2].params['b'].grad - torch_model.layer2.bias.grad.detach().numpy())
    grad_error /= 4

    if grad_error > TEST_EPS or np.isnan(grad_error):
        print(f'{grad_error=}')
        return False
    return True


@repeat
@try_except_print
def test_sequential_with_batchnorm1d_0():
    network_operations = [
        Dense(10, 5),
        Sigmoid(),
        BatchNormalization(5),
        Dense(5, 3),
        BatchNormalization(3),
        Sigmoid(),
        Dense(3, 1),
        BatchNormalization(1),
        Sigmoid()
    ]

    model = Sequential(network_operations, BinaryCrossEntropy())

    torch_operations = [
        torch.nn.Linear(10, 5),
        torch.nn.Sigmoid(),
        torch.nn.BatchNorm1d(5),  # другие параметры
        torch.nn.Linear(5, 3),
        torch.nn.BatchNorm1d(3),
        torch.nn.Sigmoid(),
        torch.nn.Linear(3, 1),
        torch.nn.BatchNorm1d(1),
        torch.nn.Sigmoid()
    ]

    with torch.no_grad():
        # dense
        for index in [0, 3, 6]:
            torch_operations[index].weight = torch.nn.Parameter(torch.from_numpy(network_operations[index].params['A'].value))
            torch_operations[index].bias = torch.nn.Parameter(torch.from_numpy(network_operations[index].params['b'].value))

        # bn
        for index in [2, 4, 7]:
            torch_operations[index].weight = torch.nn.Parameter(torch.from_numpy(network_operations[index].params['gamma'].value))
            torch_operations[index].bias = torch.nn.Parameter(torch.from_numpy(network_operations[index].params['beta'].value))

            torch_operations[index].running_mean = torch.from_numpy(network_operations[index].params['mu'].value)
            torch_operations[index].running_var = torch.from_numpy(network_operations[index].params['sigma'].value)

    _torch_model = torch.nn.Sequential(*torch_operations)
    torch_model = lambda x, y: torch_bce(_torch_model(x), y)

    # def torch_model(x, y):
    #     pred = _torch_model(x)
    #     print(pred.shape, y.shape)
    #     return torch_bce(pred, y)


    # TODO: Автоматизировать создание подобных тестов. Сделать процедуру сравнения градиентов и других характеристик

    X = np.random.random((100, 10))
    Y = 1 / (1 + np.exp(-np.sum(X, axis=1, keepdims=True)))

    torch_X = torch.from_numpy(X).cpu()
    torch_Y = torch.from_numpy(Y).cpu()

    loss = model.forward(X, Y)
    # print(np.sum(loss), 'my loss')
    model.backward()

    torch_loss = torch_model(torch_X, torch_Y)
    # print(torch_loss)
    # print(torch_loss.item(), 'torch loss')
    torch_loss.backward()

    loss_error = np.abs(np.sum(loss) - torch_loss.item())

    if loss_error > TEST_EPS or np.isnan(loss_error):
        print(f'{loss_error=}')
        return False

    grad_error = 0
    statistics_error = 0

    for index in [0, 3, 6]:
        grad_error += np.linalg.norm(torch_operations[index].weight.grad.numpy() - network_operations[index].params['A'].grad)
        grad_error += np.linalg.norm(torch_operations[index].bias.grad.numpy() - network_operations[index].params['b'].grad)

        # print('dense', index, torch_operations[index].weight.grad.numpy(), network_operations[index].params['A'].grad)

    # bn
    for index in [2, 4, 7]:
        grad_error += np.linalg.norm(torch_operations[index].weight.grad.numpy() - network_operations[index].params['gamma'].grad)
        grad_error += np.linalg.norm(torch_operations[index].bias.grad.numpy() - network_operations[index].params['beta'].grad)

        # print(index, torch_operations[index].weight.grad.numpy(), network_operations[index].params['gamma'].grad)
        # print(index, torch_operations[index].bias.grad.numpy(), network_operations[index].params['beta'].grad)
        #
        # print(index, torch_operations[index].running_mean.numpy(), network_operations[index].params['mu'].value)
        # print(index, torch_operations[index].running_var.numpy(), network_operations[index].params['sigma'].value)

        statistics_error += np.linalg.norm(torch_operations[index].running_mean.numpy() - network_operations[index].params['mu'].value)
        statistics_error += np.linalg.norm(torch_operations[index].running_var.numpy() - network_operations[index].params['sigma'].value)

    if grad_error / 12 > TEST_EPS or np.isnan(grad_error):
        print(f'{grad_error / 12=}')
        return False

    if statistics_error > TEST_EPS or np.isnan(statistics_error):
        print(f'{statistics_error=}')
        return False

    return True


def test_batchnormalization_0():
    input_size = 5

    model = Sequential([BatchNormalization(input_size)], BinaryCrossEntropy())
    torch_model = torch.nn.Sequential(torch.nn.BatchNorm1d(input_size))

    for name, param in torch_model.named_parameters():
        print(name, param)

    torch_btch_layer = list(torch_model.children())[0]
    print('torch start statistics state', torch_btch_layer.running_mean.numpy(), torch_btch_layer.running_var.numpy())

    X = np.random.random((2, input_size))
    X = np.array(X, dtype=np.float32)

    pred = model.predict(X)

    torch_X = torch.from_numpy(X).cpu().float()

    torch_pred = torch_model.forward(torch_X)

    pred_error = np.linalg.norm(pred - torch_pred.detach().numpy())


    print('my new statistics state', model.network_operations[0].params['mu'].value, model.network_operations[0].params['sigma'].value)
    torch_btch_layer = list(torch_model.children())[0]
    print('torch new statistics state', torch_btch_layer.running_mean.numpy(), torch_btch_layer.running_var.numpy())

    print(pred)
    print(torch_pred.detach().numpy())
    # print(pred - torch_pred.detach().numpy())
    # Почему-то не хочет опускаться ниже 1e-6 - 1e-7
    if pred_error > TEST_EPS or np.isnan(pred_error):
        print(f'{pred_error=}')
        return False

    return True


# TODO: Тест значений, градиентов для всех слоев во время процесса обучения на каждом шаге

########################################################################################################################


def start_testing(test_functions: List[Callable]):
    # TODO: Возвращать true/false успешен ли тест и потом считать статистики по тестам всем
    results = {}

    for test_func in test_functions:
        results[test_func.__name__] = test_func()

    success_percentage = np.average([int(is_success) for key, is_success in results.items()]) * 100

    for key, items in results.items():
        print(f'{key}: {items}')

    print(f'\n\nAll tests passes\n{success_percentage=:.1f}')


if __name__ == '__main__':
    test_functions = []

    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj) and name.startswith('test') and obj.__module__ == __name__:
            test_functions.append(obj)

    start_testing(test_functions)
