# utilis/simplification/trig_conversion.py
import sympy as sp

def trig_conversion(expr):
    """
    Konwertuje odwrotności funkcji trygonometrycznych na ich odpowiedniki.
    """
    if not isinstance(expr, sp.Basic):
        raise TypeError("Input must be a sympy expression (sp.Basic).")
    
    def convert_trig(e):
        if isinstance(e, sp.Pow) and e.exp.is_negative:
            base = e.base
            exp = abs(e.exp)
            if base.func == sp.tan:
                return sp.cot(base.args[0])**exp
            elif base.func == sp.sin:
                return sp.csc(base.args[0])**exp
            elif base.func == sp.cos:
                return sp.sec(base.args[0])**exp
            elif base.func == sp.cot:
                return sp.tan(base.args[0])**exp
            elif base.func == sp.sec:
                return sp.cos(base.args[0])**exp
            elif base.func == sp.csc:
                return sp.sin(base.args[0])**exp
        return e

    try:
        result = expr.replace(
            lambda e: isinstance(e, sp.Pow) and e.exp.is_negative,
            convert_trig
        )
        return result
    except Exception as e:
        print(f"Error in trig_conversion: {e}")
        return expr
