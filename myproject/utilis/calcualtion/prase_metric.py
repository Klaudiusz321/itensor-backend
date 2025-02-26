import sympy as sp

def wczytaj_metryke_z_tekstu(metric_text: str):
    try:
        print("\n=== Parsing metric text ===")
        print("Input text:", metric_text)
        
        if not metric_text:
            raise ValueError("Empty metric text")

        # Słownik dla symboli z dodatkowymi założeniami
        symbol_assumptions = {
            'a': dict(real=True, positive=True),
            't': dict(real=True),
            'r': dict(real=True),
            'theta': dict(real=True),
            'phi': dict(real=True),
            'chi': dict(real=True),
            'k': dict(real=True),
        }

        def create_symbol(sym_name):
            try:
                sym_name = sym_name.strip()
                if not sym_name:
                    raise ValueError("Empty symbol name")
                    
                if sym_name in symbol_assumptions:
                    return sp.Symbol(sym_name, **symbol_assumptions[sym_name])
                return sp.Symbol(sym_name)
            except Exception as e:
                print(f"Error creating symbol {sym_name}: {e}")
                raise

        lines = metric_text.split('\n')
        if not lines:
            raise ValueError("No lines in metric text")

        # Parsuj pierwszą linię dla współrzędnych
        coord_line = lines[0].strip()
        wspolrzedne = [create_symbol(s.strip()) for s in coord_line.split(',') if s.strip()]
        print("Parsed coordinates:", [str(w) for w in wspolrzedne])

        # Inicjalizuj metrykę
        n = len(wspolrzedne)
        metryka = {}
        parametry = []

        # Parsuj pozostałe linie
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            if 'g_{' in line:  # Format g_{ij} = expression
                try:
                    # Wyciągnij indeksy i wyrażenie
                    indices = line[line.find('{')+1:line.find('}')]
                    i, j = map(int, indices)
                    expr = line.split('=')[1].strip()
                    
                    # Stwórz słownik symboli
                    symbols_dict = {str(sym): sym for sym in wspolrzedne}
                    symbols_dict.update({
                        'sin': sp.sin,
                        'cos': sp.cos,
                        'tan': sp.tan,
                        'exp': sp.exp,
                        'k': sp.Symbol('k', real=True),
                        'chi': sp.Symbol('chi', real=True),
                    })
                    
                    # Dodaj funkcję a(t)
                    t = symbols_dict.get('t', sp.Symbol('t'))
                    symbols_dict['a'] = sp.Function('a')(t)
                    
                    # Parsuj wyrażenie
                    expr_sympy = sp.sympify(expr, locals=symbols_dict)
                    metryka[(i, j)] = expr_sympy
                    metryka[(j, i)] = expr_sympy  # symetria
                    
                except Exception as e:
                    print(f"Error parsing line '{line}': {e}")
                    raise ValueError(f"Invalid metric component format: {line}")

        if not metryka:
            raise ValueError("No metric components specified")

        # Zbierz wszystkie parametry (symbole, które nie są współrzędnymi)
        all_symbols = set()
        for expr in metryka.values():
            all_symbols.update(expr.free_symbols)
        parametry = list(all_symbols - set(wspolrzedne))

        print("Parsed metric components:", len(metryka))
        print("Parsed parameters:", [str(p) for p in parametry])

        return wspolrzedne, parametry, metryka

    except Exception as e:
        print(f"Error in wczytaj_metryke_z_tekstu: {str(e)}")
        raise ValueError(f"Metric parsing error: {str(e)}")
