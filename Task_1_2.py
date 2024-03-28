
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


# золотое сечение
F = (1 + math.sqrt(5)) / 2
def getStep(cur_x, cur_y, grad_x, grad_y, left, right, eps):
    count_function_runs_in_func = 0

    x1 = right - (right - left) / F
    x2 = left + (right - left) / F
    f1 = f(cur_x - x1 * grad_x, cur_y - x1 * grad_y)
    f2 = f(cur_x - x2 * grad_x, cur_y - x2 * grad_y)
    count_function_runs_in_func += 2

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

        count_function_runs_in_func += 1

    return (left + right) / 2, count_function_runs_in_func



# стартовая точка:
x_start = 2
y_start = 5
x = x_start
y = y_start

func = f(x, y)
new_func = func

learning_rate = 0.5
learning_rate_start = 0.5
eps = 0.001

# счетчик количества итераций цикла:
it = 0

# счетчик количества вызовов функции
count_function_runs = 1

# массивы, содержащие координаты точек, по которым проходим во время цикла
# для дальнейшего построения точек на графике:
X_graph = []
Y_graph = []
Z_graph = []

while True:

    X_graph.append(x)
    Y_graph.append(y)
    Z_graph.append(new_func)

    delta_f = new_func - func
    grad_x, grad_y = grad(x, y)
    delta_arg = math.sqrt(learning_rate * grad_x * learning_rate * grad_x + learning_rate * grad_y * learning_rate * grad_y)

    if it != 0 and delta_arg < eps:
        break

    func = new_func

    learning_rate, cnt_func = getStep(x, y, grad_x, grad_y, 0, learning_rate_start, eps)

    x = x - learning_rate * grad_x
    y = y - learning_rate * grad_y

    new_func = f(x, y)

    it += 1
    count_function_runs += cnt_func + 1


# конец работы программы (время):
end = time.time()

print(f'function: x ** 2 + x*y + y ** 2 - 6*x - 9*y')
print(f'start points: x = {x_start}, y = {y_start}')
print(f'learning_rate: golden({learning_rate_start}), eps: {eps}\n')
print(f'x: {x}, y: {y}\nf: {f(x, y)}, iterations: {it}\nwork time: {(end-start) * 10**3} ms')
print(f'count_function_runs: {count_function_runs}\n')


fig = plt.figure(figsize=(10, 10))
fig.set_figheight(5)

# построение точек на графике
ax = fig.add_subplot(111, projection='3d')
plt.plot(X_graph, Y_graph, Z_graph, 'r')
plt.plot(X_graph, Y_graph, Z_graph, 'bo')


# построение поверхности функции:
x = np.arange(-1, 5, 0.5)
y = np.arange(-1, 5, 0.5)
X, Y = np.meshgrid(x, y)
z = np.array(f(np.ravel(X), np.ravel(Y)))
Z = z.reshape(X.shape)

ax.plot_wireframe(X, Y, Z, cmap='viridis', edgecolor='green')
ax.set_title('Surface plot of f(x,y)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# вывод графика поверхности функции и отмеченных точек:
# (для просмотра следующего окна нужно закрыть текущее)
plt.show()

# построение линий уровня функции в окрестности точки минимума:
x = np.arange(-1, 5, 0.5)
y = np.arange(-1, 5, 0.5)
X, Y = np.meshgrid(x, y)
z = np.array(f(np.ravel(X), np.ravel(Y)))
Z = z.reshape(X.shape)

cs = plt.contour(X, Y, Z, levels=50)
plt.clabel(cs)
plt.plot(X_graph, Y_graph, 'r')
plt.plot(X_graph, Y_graph, 'bo')

# вывод линий уровня функции в окрестности точки минимума:
# (для просмотра этого окна нужно закрыть предыдущее)
plt.show()

