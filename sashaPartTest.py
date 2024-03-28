import time
import numpy as np
from scipy.optimize import minimize


class Result:
    def __init__(self, x, value, delta_time, nit, method_name, count_calc_values):
        self.x = x
        self.value = value
        self.delta_time = delta_time
        self.nit = nit
        self.method_name = method_name
        self.count_calc_values = count_calc_values


def library_nelder_mead(function, n):
    start_time = time.time()
    res = minimize(fun = function, x0 = np.random.uniform(low=0, high=1, size=n), tol = 0.0000000001, method='Nelder-Mead')
    end_time = time.time()
    return Result(res.x, function(res.x), (end_time - start_time) * 10**3, res.nit, "Nelder-Mead", res.nfev)

def create_log(data):
    print(f"Calculation method: {data.method_name}\n")
    print(f"Iterations: {data.nit}\n")
    print(f"Number of function value calculations: {data.count_calc_values}\n")
    print(f"Result args: {data.x}\n")
    print(f"Result value: {data.value}\n")
    print(f"Execution time (seconds): {data.delta_time} ms\n")

def f_Rk(x):
    return 1/100 * np.sum(x**2)

if __name__ == "__main__":
    data = library_nelder_mead(f_Rk, 30)
    create_log(data)
