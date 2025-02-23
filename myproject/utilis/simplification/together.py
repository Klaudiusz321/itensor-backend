# utilis/simplification/together_expr.py
import sympy as sp

def together_expr(expr):
    """
    Łączy wyrażenia wymierne w jeden ułamek.
    """
    if not isinstance(expr, sp.Basic):
        raise TypeError("Input must be a sympy expression (sp.Basic).")
    
    try:
        # Najpierw próbujemy połączyć ułamki
        result = sp.together(expr)
        
        # Jeśli wyrażenie jest sumą, sprawdzamy czy można je dalej uprościć
        if result.is_Add:
            num, den = sp.fraction(result)
            if den != 1:
                result = sp.cancel(num/den)
        
        return result
    except Exception as e:
        print(f"Error in together_expr: {e}")
        return expr