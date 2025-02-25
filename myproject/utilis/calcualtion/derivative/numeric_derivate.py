import numpy as np

def numeric_derivative(f, x, h=1e-5):
    """
    Liczy przybliżoną pochodną funkcji f w punkcie x
    metodą różnic centralnych.
    
    f: funkcja (Python) przyjmująca float i zwracająca float,
    x: punkt (float), w którym liczymy pochodną,
    h: mały krok (float).
    
    Zwraca: przybliżoną wartość f'(x).
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)

# Przykład użycia:
def example_function(x):
    return np.sin(x) + 2.0  # przykładowa funkcja

x0 = 1.0
print("Pochodna f'(1.0) =", numeric_derivative(example_function, x0))
