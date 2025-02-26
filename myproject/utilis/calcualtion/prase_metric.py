import sympy as sp

def wczytaj_metryke_z_tekstu(metric_text: str):
    try:
        print("\n=== Parsing metric text ===")
        print("Input text:", metric_text)
        
        symbol_assumptions = {
            'a': dict(real=True, positive=True),
            'tau': dict(real=True),
            'psi': dict(real=True),
            'theta': dict(real=True),
            'phi': dict(real=True),
        }

        def create_symbol(sym_name):
            print(f"Creating symbol: {sym_name}")
            if sym_name in symbol_assumptions:
                return sp.Symbol(sym_name, **symbol_assumptions[sym_name])
            else:
                return sp.Symbol(sym_name)

        wspolrzedne = []
        parametry = []
        metryka = {}

        lines = metric_text.splitlines()
        print("Split lines:", lines)

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
        print("Error in wczytaj_metryke_z_tekstu:", e)
        raise
