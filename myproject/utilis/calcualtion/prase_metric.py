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

        # Podziel tekst na linie i usuń puste
        lines = [line.strip() for line in metric_text.split('\n') if line.strip()]
        if not lines:
            raise ValueError("No lines in metric text")

        print("Processing lines:", lines)

        # Parsuj pierwszą linię - współrzędne
        coord_line = lines[0]
        wspolrzedne = [create_symbol(s.strip()) for s in coord_line.split(',') if s.strip()]
        print("Parsed coordinates:", [str(w) for w in wspolrzedne])

        # Inicjalizuj metrykę
        n = len(wspolrzedne)
        metryka = {}

        # Stwórz bazowy słownik symboli
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
        a_func = sp.Function('a')(t)
        symbols_dict['a'] = a_func

        # Parsuj komponenty metryki
        for line in lines[1:]:
            if '=' in line:  # To jest linia z komponentą metryki
                try:
                    # Znajdź indeksy
                    if 'g_{' in line:
                        indices = line[line.find('{')+1:line.find('}')]
                        i, j = map(int, indices)
                    else:
                        # Alternatywny format: g_ij lub g(i,j)
                        for char in '()_{}':
                            line = line.replace(char, ' ')
                        parts = line.split('=')[0].split()
                        i, j = map(int, [p for p in parts if p.isdigit()])

                    # Pobierz wyrażenie
                    expr = line.split('=')[1].strip()
                    
                    # Parsuj wyrażenie
                    expr_sympy = sp.sympify(expr, locals=symbols_dict)
                    metryka[(i, j)] = expr_sympy
                    metryka[(j, i)] = expr_sympy  # symetria
                    print(f"Added metric component ({i},{j}): {expr}")
                    
                except Exception as e:
                    print(f"Error parsing line '{line}': {e}")
                    raise ValueError(f"Invalid metric component format: {line}")

        if not metryka:
            raise ValueError("No metric components specified")

        # Zbierz wszystkie parametry
        all_symbols = set()
        for expr in metryka.values():
            all_symbols.update(expr.free_symbols)
        parametry = list(all_symbols - set(wspolrzedne))

        print("Final metric components:", len(metryka))
        print("Final parameters:", [str(p) for p in parametry])
        print("Metric:", metryka)

        return wspolrzedne, parametry, metryka

    except Exception as e:
        print(f"Error in wczytaj_metryke_z_tekstu: {str(e)}")
        raise ValueError(f"Metric parsing error: {str(e)}")
