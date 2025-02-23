# utilis/simplification/pow_simplify.py
import sympy as sp

def pow_simplify(expr, deep=True, force=False):
   
    if not isinstance(expr, sp.Basic):
        raise TypeError("Input must be a sympy expression (sp.Basic).")
    try:
        result = sp.powsimp(expr, deep=deep, force=force)
        return result
    except Exception as e:
        print(f"Error in pow_simplify: {e}")
        return expr
