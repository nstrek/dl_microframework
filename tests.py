import sys
import inspect
from framework import Parameters, Tensor, np
from typing import List, Callable


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
