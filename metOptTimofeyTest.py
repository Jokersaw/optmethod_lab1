
import math
import time

import numpy as np
import matplotlib.pyplot as plt


# начало работы программы (время):
start = time.time()

def grad(x, y):
    return -2 + 2*x + 400*x**3 - 200*y, 100 - 200*x**2


# основная функция:
def f(x, y):
    return (1-x)**2+100*(y-x**2)**2


#f in R^k
def f_Rk(x):
    return np.sum(x**2)

def grad_Rk(x):
    return 2*x


# условие останова:
# (2 - малое приращение аргумента)


# стартовая точка:
x = 0.8
y = 1

# Задаем размерность вектора n
n = 2
random_npoint = np.array

#### saveeee
save = np.array
####

func = 0
if n==2:
    func = f(x, y)
else:
    random_npoint = np.random.uniform(low=0, high=5, size=n)
    save = random_npoint
    func = f_Rk(random_npoint)
new_func = func    

step = 0.005
eps = 0.00001

# счетчик количества итераций цикла:
it = 0

# массивы, содержащие координаты точек, по которым проходим во время цикла
# для дальнейшего построения точек на графике:
X_graph = []
Y_graph = []
Z_graph = []

delta_f = 0
grad_vec = np.array
norma_delta_arg = 0
point = np.array

while True:
    if n == 2:
        X_graph.append(x)
        Y_graph.append(y)
        Z_graph.append(func)

        delta_f = new_func - func
        grad_x, grad_y = grad(x, y)
        norma_delta_arg = math.sqrt(step * grad_x * step * grad_x + step * grad_y * step * grad_y)
        if it != 0 and norma_delta_arg < eps:
            break

        func = new_func

        x = x - step * grad_x
        y = y - step * grad_y

        new_func = f(x, y)
        grad_vec = [grad_x, grad_y]
        point = [x,y]
    else:
        delta_f = new_func - func
        grad_x = grad_Rk(random_npoint)
        norma_delta_arg = np.linalg.norm(step * grad_x)
        if it != 0 and norma_delta_arg < eps:  # Критерий останова
            break
        func = new_func
        random_npoint = random_npoint - step * grad_x
        new_func = f_Rk(random_npoint)
        grad_vec = grad_x
        point = random_npoint
    
    it += 1

# конец работы программы (время):
end = time.time()

print(f'n={n}')
print(f'a_k: {step}, eps: {eps}\n')
print(f'point: {point}\nf: {func}, iterations: {it}\nwork time: {(end-start) * 10**3} ms\n')
print(f'delta_f: {delta_f}\nnorma_delta_argument: {norma_delta_arg}\n')
print(f'nstartpoint: {save}')


if n == 2:
    fig = plt.figure(figsize=(10, 10))
    fig.set_figheight(5)

    # построение точек на графике
    # (точка C(1;0.5) - красного цвета
    # точки, полученные во время работы программы - синего):
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_graph, Y_graph, Z_graph)
    ax.scatter(1, 0.5, 0.25, c=["red"])

    # построение поверхности функции:
    x = np.arange(-4, 6, 0.05)
    y = np.arange(-1, 9, 0.05)
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

    # построение линий уровня функции в окрестности точки C:
    x = np.arange(-4, 6, 0.8)
    y = np.arange(-1, 9, 0.8)
    X, Y = np.meshgrid(x, y)
    z = np.array(f(np.ravel(X), np.ravel(Y)))
    Z = z.reshape(X.shape)

    # cs = plt.contour(X, Y, Z, levels=50)
    cs = plt.contour(X, Y, Z, levels=50)
    plt.clabel(cs)
    plt.plot(X_graph, Y_graph, 'ro')

    # вывод линий уровня функции в окрестности точки C:
    # (для просмотра этого окна нужно закрыть предыдущее)
    plt.show()
