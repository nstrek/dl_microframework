import sys
import inspect
from framework import Parameters, Tensor, np, Dense
from typing import List, Callable
import torch


def test_parameters_0():
    try:
        p = Parameters()

        p['5'] = Tensor(np.zeros(1))

        try:
            p[5] = Tensor(np.ones(1))
            return False
        except ValueError:
            pass
    except Exception:
        return False

    return True


def test_dense_derivative_0():
    try:
        dense = Dense(10, 2)

        X = Tensor(np.random.random((3, 10)))

        pred = dense.forward(X)['df']

        print(pred.shape)
        print(pred)
        print(np.sum(pred, axis=0))

        torch_A = torch.from_numpy(dense.params['A'].value)
        torch_A.requires_grad_()
        torch_X = torch.from_numpy(np.expand_dims(X.value, axis=2))

        f = torch.sum(torch.matmul(torch_A, torch_X))
        f.backward()

        if np.any(torch_A.grad.numpy() != A_grad):
            return False
    except Exception:
        return False

    return True


def start_testing(test_functions: List[Callable]):
    # TODO: Возвращать true/false успешен ли тест и потом считать статистики по тестам всем
    results = {}

    for test_func in test_functions:
        results[test_func.__name__] = test_func()

    success_percentage = np.average([int(is_success) for key, is_success in results.items()]) * 100

    print(f'All tests passes\n{success_percentage=:.1f}')


if __name__ == '__main__':
    test_functions = []

    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj) and name.startswith('test') and obj.__module__ == __name__:
            test_functions.append(obj)

    start_testing(test_functions)
