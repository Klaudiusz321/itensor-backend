import sympy as sp

def wczytaj_metryke_z_tekstu(metric_text: str):
    symbol_assumptions = {
        'a':    dict(real=True, positive=True),
        'tau':  dict(real=True),
        'psi':  dict(real=True),
        'theta':dict(real=True),
        'phi':  dict(real=True),
    }

    def create_symbol(sym_name):
        if sym_name in symbol_assumptions:
            return sp.Symbol(sym_name, **symbol_assumptions[sym_name])
        else:
            return sp.Symbol(sym_name)

    wspolrzedne = []
    parametry = []
    metryka = {}

    # Dzielimy tekst na linie
    lines = metric_text.splitlines()

    for line in lines:
        # Usuwamy komentarze i puste linie
        line = line.split('#')[0].strip()
        if not line:
            continue

        if ';' in line:
            wsp_, prm_ = line.split(';')
            wsp_strs = [sym.strip() for sym in wsp_.split(',') if sym.strip()]
            wspolrzedne = [create_symbol(s) for s in wsp_strs]

            prm_ = prm_.strip()
            if prm_:
                par_strs = [sym.strip() for sym in prm_.split(',') if sym.strip()]
                parametry = [create_symbol(s) for s in par_strs]
        else:
            dat = line.split(maxsplit=2)
            if len(dat) == 3:
                try:
                    i, j, expr = int(dat[0]), int(dat[1]), dat[2]
                    symbols_dict = {str(sym): sym for sym in wspolrzedne + parametry}
                    metryka[(i, j)] = sp.sympify(expr, locals=symbols_dict)
                except ValueError:
                    print(f"Error: Incorrect data in line: {line}")
    return wspolrzedne, parametry, metryka
