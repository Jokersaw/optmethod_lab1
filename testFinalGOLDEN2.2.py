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
    print(f"Calculation method: {data.method_name}\n")
    print(f"Computed function: {data.func}\n")
    print(f"Iterations: {data.nit}\n")
    print(f"Number of function value calculations: {data.count_calc_values}\n")
    print(f"Result args: {data.x}\n")
    print(f"Result value: {data.value}\n")
    print(f"Execution time (seconds): {data.delta_time}\n")


if __name__ == "__main__":
    f = Function(lambda x: (1-x[0])**2+100*(x[1]-x[0]**2)**2, 3, "(1-x)**2+100*(y-x**2)**2")
    data = library_nelder_mead(f)
    create_log(data, "library_nelder_mead.log")

    
