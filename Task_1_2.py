
import math
import time

import numpy as np
import matplotlib.pyplot as plt


# начало работы программы (время):
start = time.time()

def grad(x, y):
    return 2*x + y - 6, x + 2*y - 9


# основная функция:
def f(x, y):
    return x ** 2 + x*y + y ** 2 - 6*x - 9*y


F = (1 + math.sqrt(5)) / 2
def getStep(cur_x, cur_y, grad_x, grad_y, left, right, eps):
    x1 = right - (right - left) / F
    x2 = left + (right - left) / F
    f1 = f(cur_x - x1 * grad_x, cur_y - x1 * grad_y)
    f2 = f(cur_x - x2 * grad_x, cur_y - x2 * grad_y)

    while abs(right - left) >= eps:

        if f1 >= f2:
            left = x1
            x1, f1 = x2, f2
            x2 = left + (right - left) / F
            f2 = f(cur_x - x2 * grad_x, cur_y - x2 * grad_y)
        else:
            right = x2
            x2, f2 = x1, f1
            x1 = right - (right - left) / F
            f1 = f(cur_x - x1 * grad_x, cur_y - x1 * grad_y)

    return (left + right) / 2




# условие останова:
# (1 - малое изменение значения функции)
# (2 - малое приращение аргумента)
type = 2


# стартовая точка:
x = 0.8
y = 1

func = f(x, y)
new_func = func


step = 0.005
eps = 0.00001

# счетчик количества итераций цикла:
it = 0

# массивы, содержащие координаты точек, по которым проходим во время цикла
# для дальнейшего построения точек на графике:
X = []
Y = []
Z = []

while True:

    X.append(x)
    Y.append(y)
    Z.append(func)

    delta_f = new_func - func
    grad_x, grad_y = grad(x, y)
    norma_delta_arg = math.sqrt(step * grad_x * step * grad_x + step * grad_y * step * grad_y)

    # 1 условие останова (малое приращение функции: приращение функции < eps):
    if type == 1 and it != 0 and abs(delta_f) < eps:
        break

    # 2 условие останова (малое приращение аргумента: норма приращения < eps):
    if type == 2 and it != 0 and norma_delta_arg < eps:
        break

    func = new_func

    step = getStep(x, y, grad_x, grad_y, 0, 0.05, eps)

    x = x - step * grad_x
    y = y - step * grad_y

    new_func = f(x, y)

    it += 1

# конец работы программы (время):
end = time.time()

print(f'step: golden, eps: {eps}\n')
print(f'x: {x}, y: {y}\nf: {func}, iterations: {it}\nwork time: {(end-start) * 10**3} ms\n')
print(f'delta_f: {delta_f}\nnorma_delta_argument: {norma_delta_arg}')






