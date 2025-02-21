import sympy as sp

def custom_simplify(expr):
    """
    Uproszczenie wyrażenia symbolicznego.
    """
    try:
        # Uproszczenie wyrażenia
        expr_simpl = sp.expand(expr)
        expr_simpl = sp.trigsimp(expr_simpl)
        expr_simpl = sp.factor(expr_simpl)
        expr_simpl = sp.simplify(expr_simpl)
        expr_simpl = sp.cancel(expr_simpl)
        expr_simpl = sp.ratsimp(expr_simpl)

        # Dodatkowe uproszczenia
        expr_simpl = sp.logcombine(expr_simpl, force=True)  # Uproszczenie logarytmów
        expr_simpl = sp.simplify(expr_simpl, rational=True)  # Uproszczenie wyrażeń racjonalnych

        # Uproszczenie trygonometryczne
        expr_simpl = sp.trigsimp(expr_simpl)

        # Zamiana 1/tan(x) na cot(x)
        expr_simpl = expr_simpl.replace(1/sp.tan(sp.Symbol('x')), sp.cot(sp.Symbol('x')))

        return expr_simpl
    except Exception as e:
        print(f"Error in custom_simplify: {e}")
        return expr  # Zwracamy oryginalne wyrażenie w przypadku błędu 