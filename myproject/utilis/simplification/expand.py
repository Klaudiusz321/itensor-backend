# utilis/simplification/expand_expr.py
import sympy as sp

def expand_expr(expr, deep=False, mul=False, power_base=False, force=False):
  
    if not isinstance(expr, sp.Basic):
        raise TypeError("Input must be a sympy expression (sp.Basic).")
    
    try:
        expanded_expr = sp.expand(expr, deep=deep, mul=mul, power_base=power_base)
        
        # Jeśli nie wymuszamy i rozwinięcie nie zmieniło wyrażenia, zwracamy oryginał
        if not force and expanded_expr == expr:
            return expr
        return expanded_expr
    except Exception as e:
        print(f"Error in expand_expr: {e}")
        return expr
