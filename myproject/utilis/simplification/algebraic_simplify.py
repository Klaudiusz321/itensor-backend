import sympy as sp

def algebraic_simplify(expr, rational=True):
    """
    Wykonuje serię algebraicznych uproszczeń.
    """
    if not isinstance(expr, sp.Basic):
        raise TypeError("Input must be a sympy expression (sp.Basic).")
    try:
        # Kolejność uproszczeń jest ważna
        result = expr
        
        # Łączenie podobnych wyrazów
        if result.is_Add:
            terms = {}  # słownik do grupowania podobnych wyrazów
            
            for term in result.args:
                # Rozdzielamy współczynnik i resztę wyrażenia
                coeff = 1
                base = term
                
                # Jeśli term jest iloczynem (np. 2*x)
                if term.is_Mul:
                    coeffs = []
                    non_coeffs = []
                    for factor in term.args:
                        if factor.is_number:
                            coeffs.append(factor)
                        else:
                            non_coeffs.append(factor)
                    if coeffs:
                        coeff = sp.Mul(*coeffs)
                        base = sp.Mul(*non_coeffs) if non_coeffs else 1
                    else:
                        base = term
                # Jeśli term jest liczbą
                elif term.is_number:
                    coeff = term
                    base = 1
                
                # Dodajemy do słownika
                if base in terms:
                    terms[base] += coeff
                else:
                    terms[base] = coeff
            
            # Składamy wyrażenie z powrotem
            result = 0
            for base, coeff in terms.items():
                if coeff != 0:  # pomijamy wyrazy z zerowym współczynnikiem
                    if base == 1:
                        result += coeff
                    elif coeff == 1:
                        result += base
                    else:
                        result += coeff * base
        
        # Standardowe uproszczenia
        result = sp.cancel(result)  # Upraszcza ułamki
        result = sp.collect(result, result.free_symbols)  # Grupuje podobne terminy
        result = sp.factor_terms(result)  # Wyciąga wspólne czynniki
        result = sp.simplify(result, rational=rational)  # Ogólne uproszczenie
        result = sp.ratsimp(result)  # Upraszcza wyrażenia wymierne
        
        return result
    except Exception as e:
        print(f"Error in algebraic_simplify: {e}")
        return expr