import sympy as sp


def custom_simplify(expr):
    """
    Maksymalnie upraszcza wyrażenie symboliczne.
    """
    try:
        # Rozwijanie wyrażenia i wstępne uproszczenia
        expr_simpl = sp.expand(expr)
        expr_simpl = sp.trigsimp(expr_simpl)
        expr_simpl = sp.logcombine(expr_simpl, force=True)
        expr_simpl = sp.powsimp(expr_simpl, deep=True)
        expr_simpl = sp.together(expr_simpl)
        
        # Uproszczenia algebraiczne
        expr_simpl = sp.cancel(expr_simpl)
        expr_simpl = sp.ratsimp(expr_simpl)
        expr_simpl = sp.factor_terms(expr_simpl)
        expr_simpl = sp.simplify(expr_simpl, rational=True)
        
        # Kolejny przebieg uproszczenia trygonometrycznego
        expr_simpl = sp.trigsimp(expr_simpl)
        
        # Zamiana wyrażeń trygonometrycznych
        def convert_trig(e):
            if isinstance(e, sp.Pow):
                # Zamiana 1/tan(x) na cot(x)
                if e.base.func == sp.tan and e.exp == -1:
                    return sp.cot(e.base.args[0])
                # Zamiana 1/sin(x) na csc(x)
                elif e.base.func == sp.sin and e.exp == -1:
                    return sp.csc(e.base.args[0])
                # Zamiana 1/cos(x) na sec(x)
                elif e.base.func == sp.cos and e.exp == -1:
                    return sp.sec(e.base.args[0])
            return e

        expr_simpl = expr_simpl.replace(
            lambda e: isinstance(e, sp.Pow) and e.exp == -1,
            convert_trig
        )
        
        # Wymuszenie ujemnych wykładników
     
        
        return expr_simpl
    except Exception as e:
        print(f"Error in custom_simplify: {e}")
        return expr
