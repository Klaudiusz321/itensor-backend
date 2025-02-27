import sympy as sp

def wczytaj_metryke_z_tekstu(metric_text: str):
    try:
        print("\n=== Parsing metric text ===")
        print("Input text:", metric_text)
        
        # Słownik dla symboli z dodatkowymi założeniami
        symbol_assumptions = {
            'a': dict(real=True, positive=True),
            't': dict(real=True),
            'tau': dict(real=True),
            'psi': dict(real=True),
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

        wspolrzedne = []
        parametry = []
        metryka = {}

        # Podziel tekst na linie i usuń komentarze i puste linie
        lines = [line.split('#')[0].strip() for line in metric_text.split('\n')]
        lines = [line for line in lines if line]

        if not lines:
            raise ValueError("No valid lines in input")

        # Parsuj pierwszą linię - współrzędne i parametry
        first_line = lines[0]
        if ';' in first_line:
            wsp_, prm_ = first_line.split(';')
            # Parsuj współrzędne
            wsp_strs = [sym.strip() for sym in wsp_.split(',') if sym.strip()]
            wspolrzedne = [create_symbol(s) for s in wsp_strs]
            
            # Parsuj parametry
            prm_ = prm_.strip()
            if prm_:
                par_strs = [sym.strip() for sym in prm_.split(',') if sym.strip()]
                parametry = [create_symbol(s) for s in par_strs]
        else:
            raise ValueError("First line must contain coordinates and parameters separated by ';'")

        print("Parsed coordinates:", [str(w) for w in wspolrzedne])
        print("Parsed parameters:", [str(p) for p in parametry])

        # Stwórz słownik symboli dla parsowania wyrażeń
        symbols_dict = {str(sym): sym for sym in wspolrzedne + parametry}
        symbols_dict.update({
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'exp': sp.exp,
            'M': sp.Symbol('M', real=True, positive=True)
        })

        # Parsuj komponenty metryki
        for line in lines[1:]:
            if 'g_{' in line and '=' in line:
                # Format g_{ij} = expr
                try:
                    indices = line[line.find('{')+1:line.find('}')]
                    i, j = map(int, indices)
                    expr = line.split('=')[1].strip()
                    expr_sympy = sp.sympify(expr, locals=symbols_dict)
                    metryka[(i, j)] = expr_sympy
                    metryka[(j, i)] = expr_sympy  # symetria
                    print(f"Added metric component ({i},{j}): {expr}")
                except Exception as e:
                    print(f"Error parsing line '{line}': {e}")
                    raise ValueError(f"Invalid metric component format: {line}")
            else:
                # Format: i j expr
                try:
                    parts = line.split(maxsplit=2)
                    if len(parts) == 3:
                        i, j, expr = int(parts[0]), int(parts[1]), parts[2]
                        expr_sympy = sp.sympify(expr, locals=symbols_dict)
                        metryka[(i, j)] = expr_sympy
                        metryka[(j, i)] = expr_sympy  # symetria
                        print(f"Added metric component ({i},{j}): {expr}")
                except Exception as e:
                    print(f"Error parsing line '{line}': {e}")
                    continue

        # Uzupełnij brakujące komponenty metryki
        n = len(wspolrzedne)
        full_metric = {}
        for i in range(n):
            for j in range(n):
                if (i, j) in metryka:
                    full_metric[(i, j)] = metryka[(i, j)]
                elif (j, i) in metryka:
                    full_metric[(i, j)] = metryka[(j, i)]
                else:
                    # Dla metryki diagonalnej, zakładamy 0 dla elementów pozadiagonalnych
                    full_metric[(i, j)] = 0 if i != j else 1

        print("Final metric components:", len(full_metric))
        print("Full metric:", full_metric)

        return wspolrzedne, parametry, full_metric

    except Exception as e:
        print(f"Error in wczytaj_metryke_z_tekstu: {str(e)}")
        raise ValueError(f"Metric parsing error: {str(e)}")
