import time
import numpy as np
from library_nelder import Result, create_log, Function


def my_nelder_mead(func, converge_by_iter=False, eps=10 ** -9, alpha=1, beta=0.5, gamma=2, max_count_iterations=1_000):
    start_time = time.time()
    fn = func.fn
    points = np.random.rand(func.dimension, func.dimension - 1)
    cur_iter = 0
    count_calc_values = 0
    while True:
        values = np.array([fn(p) for p in points])
        indexes = values.argsort()
        points = points[indexes]
        values.sort()
        if converge_by_iter and cur_iter == max_count_iterations:
            break
        elif not converge_by_iter and np.allclose(values[0], values, atol=eps):
            break
        cur_iter += 1

        mid = np.mean(points[:-1], 0)
        # reflection
        xr = mid + alpha * (mid - points[-1])
        xr_value = fn(xr)
        count_calc_values += 1
        if values[0] < xr_value < values[-1]:
            points[-1] = xr
            continue
        # expansion
        if xr_value < values[0]:
            xe = mid + gamma * (xr - mid)
            xe_value = fn(xe)
            count_calc_values += 1
            points[-1] = xe if xe_value < xr_value else xr
            continue
        # contraction or shrinkage
        xc = mid - beta * (mid - points[-1])
        xc_value = fn(xc)
        count_calc_values += 1
        if xc_value < values[-1]:
            points[-1] = xc
        else:
            points[1:] = points[0] + (points[1:] - points[0]) / 2

    end_time = time.time()
    return Result(points[0], fn(points[0]), end_time - start_time, cur_iter, "Nelder-Mead",
                  func.string_format, count_calc_values)


if __name__ == "__main__":
    f = Function(lambda x: x[0] ** 2 + x[0] * x[1] + x[1] ** 2 - 6 * x[0] - 9 * x[1], 3, "x**2+x*y+y**2-6*x-9*y")
    data = my_nelder_mead(f)
    create_log(data, "my_nelder_mead.log")
