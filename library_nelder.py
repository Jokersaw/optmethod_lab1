import time
import numpy as np
from scipy.optimize import minimize


class Function:
    def __init__(self, fn, dimension, string_format):
        self.fn = fn
        self.dimension = dimension
        self.string_format = string_format


class Result:
    def __init__(self, x, value, delta_time, nit, method_name, func, count_calc_values):
        self.x = x
        self.value = value
        self.delta_time = delta_time
        self.nit = nit
        self.method_name = method_name
        self.func = func
        self.count_calc_values = count_calc_values


def library_nelder_mead(function):
    start_time = time.time()
    res = minimize(function.fn, np.array([1, 0]), method='Nelder-Mead')
    end_time = time.time()
    return Result(res.x, function.fn(res.x), end_time - start_time, res.nit, "Nelder-Mead",
                  function.string_format, res.nfev)


def create_log(data, file_name):
    with open(file_name, 'w') as file:
        file.write(f"Calculation method: {data.method_name}\n")
        file.write(f"Computed function: {data.func}\n")
        file.write(f"Iterations: {data.nit}\n")
        file.write(f"Number of function value calculations: {data.count_calc_values}\n")
        file.write(f"Result args: {data.x}\n")
        file.write(f"Result value: {data.value}\n")
        file.write(f"Execution time (seconds): {data.delta_time}\n")


if __name__ == "__main__":
    f = Function(lambda x: x[0] ** 2 + x[0] * x[1] + x[1] ** 2 - 6 * x[0] - 9 * x[1], 3, "x**2+x*y+y**2-6*x-9*y")
    data = library_nelder_mead(f)
    create_log(data, "library_nelder_mead.log")
