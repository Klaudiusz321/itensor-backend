import numpy as np
from .numeric_derivate import numeric_derivative


def total_derivative(f, x_func, y_func, t, h=1e-5):
    """
    Liczy przybliżoną całkowitą pochodną funkcji f(x, y) względem t, gdzie x i y zależą od t.
    
    x_func, y_func: funkcje zwracające wartości x(t) i y(t).
    t: punkt, w którym liczymy pochodną.
    """
    # Najpierw oblicz wartości x i y w punkcie t
    x_val = x_func(t)
    y_val = y_func(t)
    
    # Definiujemy f_t jako funkcję f(x(t), y(t))
    def f_t(t_val):
        return f(np.array([x_func(t_val), y_func(t_val)]))
    
    # Liczymy przybliżoną pochodną f_t względem t
    return numeric_derivative(f_t, t, h)

# Przykład:
def x_func(t):
    return np.sin(t)

def y_func(t):
    return np.cos(t)

def f_xy(x_arr):
    # Przykładowa funkcja: f(x, y) = x * y
    return x_arr[0] * x_arr[1]

t0 = 1.0
print("Całkowita pochodna f względem t:", total_derivative(f_xy, x_func, y_func, t0))

