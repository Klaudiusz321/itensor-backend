# utilis/simplification/powdenest_expr.py
import sympy as sp

def powdenest_expr(expr, force=True):
   
    if not isinstance(expr, sp.Basic):
        raise TypeError("Input must be a sympy expression (sp.Basic).")
    try:
        result = sp.powdenest(expr, force=force)
        return result
    except Exception as e:
        print(f"Error in powdenest_expr: {e}")
        return expr
