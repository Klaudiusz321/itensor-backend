# utilis/simplification/trig_simplify.py
import sympy as sp

def trig_simplify(expr, method='fu'):
    """
    Upraszcza wyrażenia trygonometryczne używając wzorów redukcyjnych.
    
    Parametry:
    ----------
    expr : sp.Basic
        Wyrażenie do uproszczenia
    method : str, optional
        Metoda upraszczania ('fu', 'combined' lub 'basic')
    """
    if not isinstance(expr, sp.Basic):
        raise TypeError("Input must be a sympy expression (sp.Basic).")
    
    try:
        # Podstawowe uproszczenie
        result = expr
        
        # Wzory redukcyjne dla kąta podwójnego
        trig_double_angle = {
            sp.sin(2*sp.Wild('alpha')): 2*sp.sin(sp.Wild('alpha'))*sp.cos(sp.Wild('alpha')),
            sp.cos(2*sp.Wild('alpha')): sp.cos(sp.Wild('alpha'))**2 - sp.sin(sp.Wild('alpha'))**2,
            sp.cos(2*sp.Wild('alpha')): 2*sp.cos(sp.Wild('alpha'))**2 - 1,
            sp.cos(2*sp.Wild('alpha')): 1 - 2*sp.sin(sp.Wild('alpha'))**2,
            sp.tan(2*sp.Wild('alpha')): 2*sp.tan(sp.Wild('alpha'))/(1 - sp.tan(sp.Wild('alpha'))**2)
        }
        
        # Próbujemy zastosować każdy wzór i wybieramy najkrótszą formę
        for pattern, replacement in trig_double_angle.items():
            new_expr = result.replace(pattern, replacement)
            # Jeśli nowa forma jest krótsza lub równa długością, ale prostsza
            if len(str(new_expr)) <= len(str(result)):
                result = new_expr
        
        # Standardowe uproszczenie trygonometryczne
        if method == 'fu':
            result = sp.trigsimp(result)
        elif method == 'combined':
            result = sp.trigsimp(result, method='combined')
        else:  # 'basic'
            result = sp.simplify(result)
        
        # Dodatkowe uproszczenia dla funkcji trygonometrycznych
        if result.has(sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc):
            # Próba zamiany odwrotności funkcji trygonometrycznych
            result = result.replace(
                lambda x: isinstance(x, sp.Pow) and x.exp == -1,
                lambda x: {
                    sp.sin: sp.csc,
                    sp.cos: sp.sec,
                    sp.tan: sp.cot,
                    sp.cot: sp.tan,
                    sp.sec: sp.cos,
                    sp.csc: sp.sin
                }.get(x.base.func, lambda y: x.base**-1)(x.base.args[0])
                if isinstance(x.base, sp.Function) else x
            )
            
            # Końcowe uproszczenie
            result = sp.simplify(result)
            result = sp.trigsimp(result)
        
        return result
    except Exception as e:
        print(f"Error in trig_simplify: {e}")
        return expr
