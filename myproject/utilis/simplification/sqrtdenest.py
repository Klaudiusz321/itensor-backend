# utilis/simplification/sqrtdenest_expr.py
import sympy as sp

def sqrtdenest_expr(expr):
  
    if not isinstance(expr, sp.Basic):
        raise TypeError("Input must be a sympy expression (sp.Basic).")
    try:
        result = sp.sqrtdenest(expr)
        return result
    except Exception as e:
        print(f"Error in sqrtdenest_expr: {e}")
        return expr
