import sympy as sp

def wczytaj_metryke_z_tekstu(metric_text: str):
    try:
        if not metric_text or not isinstance(metric_text, str):
            raise ValueError("Invalid metric_text input")

        print("\n=== Parsing metric text ===")
        print("Input text:", metric_text)
        
        symbol_assumptions = {
            'a': dict(real=True, positive=True),
            'tau': dict(real=True),
            'psi': dict(real=True),
            'theta': dict(real=True),
            'phi': dict(real=True),
            't': dict(real=True),
            'r': dict(real=True),
            'x': dict(real=True),
            'y': dict(real=True),
            'z': dict(real=True),
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

        if not metric_text.strip():
            raise ValueError("Empty metric text")

        lines = [line.strip() for line in metric_text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("No valid lines in metric text")

        print("Processing lines:", lines)

        # Pierwsza linia musi zawierać współrzędne
        first_line = lines[0]
        if ';' not in first_line:
            raise ValueError("First line must contain coordinates (format: x,y,z;a,b)")

        for line in lines:
            line = line.split('#')[0].strip()
            if not line:
                continue

            print("Processing line:", line)
            
            if ';' in line:
                wsp_, prm_ = line.split(';')
                wsp_strs = [sym.strip() for sym in wsp_.split(',') if sym.strip()]
                wspolrzedne = [create_symbol(s) for s in wsp_strs]
                print("Parsed coordinates:", [str(w) for w in wspolrzedne])

                prm_ = prm_.strip()
                if prm_:
                    par_strs = [sym.strip() for sym in prm_.split(',') if sym.strip()]
                    parametry = [create_symbol(s) for s in par_strs]
                    print("Parsed parameters:", [str(p) for p in parametry])
            else:
                try:
                    dat = line.split(maxsplit=2)
                    if len(dat) == 3:
                        i, j, expr = int(dat[0]), int(dat[1]), dat[2]
                        symbols_dict = {str(sym): sym for sym in wspolrzedne + parametry}
                        metryka[(i, j)] = sp.sympify(expr, locals=symbols_dict)
                        print(f"Added metric component ({i},{j}): {expr}")
                except ValueError as e:
                    print(f"Error parsing line '{line}': {e}")
                    raise ValueError(f"Invalid metric format in line: {line}")

        if not wspolrzedne:
            raise ValueError("No coordinates specified")
        if not metryka:
            raise ValueError("No metric components specified")

        return wspolrzedne, parametry, metryka

    except Exception as e:
        print(f"Error in wczytaj_metryke_z_tekstu: {str(e)}")
        raise ValueError(f"Metric parsing error: {str(e)}")
