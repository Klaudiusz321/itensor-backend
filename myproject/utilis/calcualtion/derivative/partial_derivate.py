import numpy as np

def partial_derivative(f, x, i, h=1e-5):
    """
    Liczy przybliżoną pochodną cząstkową funkcji f w punkcie x
    względem zmiennej x[i], metodą różnic centralnych.
    
    f: funkcja, która przyjmuje np. numpy.array i zwraca float.
    x: numpy.array, punkt, w którym liczymy pochodną.
    i: indeks zmiennej, względem której liczymy pochodną.
    h: mały krok.
    
    Zwraca: przybliżoną wartość ∂f/∂x_i.
    """
    x_forward = np.copy(x)
    x_backward = np.copy(x)
    
    x_forward[i] += h
    x_backward[i] -= h
    
    return (f(x_forward) - f(x_backward)) / (2.0 * h)

# Przykład użycia dla funkcji dwóch zmiennych:
def f_multi(x):
    # Przykładowa funkcja: f(x, y) = sin(x) * cos(y)
    return np.sin(x[0]) * np.cos(x[1])

point = np.array([1.0, 1.0])
# Obliczamy ∂f/∂x dla x[0]
print("Pochodna cząstkowa względem x[0]:", partial_derivative(f_multi, point, 0))
# Obliczamy ∂f/∂y dla x[1]
print("Pochodna cząstkowa względem x[1]:", partial_derivative(f_multi, point, 1))
