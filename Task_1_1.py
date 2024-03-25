
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


# условие останова:
# (1 - малое изменение значения функции)
# (2 - малое приращение аргумента)
type = 2


# стартовая точка:
x = 0.8
y = 1

func = f(x, y)
new_func = func


a = 0.005
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
    norma_delta_arg = math.sqrt(a * grad_x * a * grad_x + a * grad_y * a * grad_y)

    # 1 условие останова (малое приращение функции: приращение функции < eps):
    if type == 1 and it != 0 and abs(delta_f) < eps:
        break

    # 2 условие останова (малое приращение аргумента: норма приращения < eps):
    if type == 2 and it != 0 and norma_delta_arg < eps:
        break

    func = new_func

    x = x - a * grad_x
    y = y - a * grad_y

    new_func = f(x, y)

    it += 1

# конец работы программы (время):
end = time.time()

print(f'a_k: {a}, eps: {eps}\n')
print(f'x: {x}, y: {y}\nf: {func}, iterations: {it}\nwork time: {(end-start) * 10**3} ms\n')
print(f'delta_f: {delta_f}\nnorma_delta_argument: {norma_delta_arg}')






