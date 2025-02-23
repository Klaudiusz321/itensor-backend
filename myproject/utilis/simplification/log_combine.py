# utilis/simplification/log_combine.py
import sympy as sp

def log_combine(expr, force=True):
    
    if not isinstance(expr, sp.Basic):
        raise TypeError("Input must be a sympy expression (sp.Basic).")
    try:
        result = sp.logcombine(expr, force=force)
        return result
    except Exception as e:
        print(f"Error in log_combine: {e}")
        return expr
