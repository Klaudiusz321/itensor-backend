from functools import lru_cache
import sympy as sp

def wczytaj_metryke_z_tekstu(metric_text: str):
    symbol_assumptions = {
        'a': dict(real=True, positive=True),
        'tau': dict(real=True),
        'psi': dict(real=True),
        'theta': dict(real=True),
        'phi': dict(real=True),
        'chi': dict(real=True),
        'k': dict(real=True)
    }

    def create_symbol(sym_name):
        sym_name = sym_name.strip()
        if sym_name in symbol_assumptions:
            return sp.Symbol(sym_name, **symbol_assumptions[sym_name])
        return sp.Symbol(sym_name)

    wspolrzedne = []
    parametry = []
    metryka = {}

    # Podziel na linie i usuń komentarze
    lines = [line.split('#')[0].strip() for line in metric_text.split('\n')]
    lines = [line for line in lines if line]

    if not lines:
        raise ValueError("Empty metric text")

    # Parsuj pierwszą linię
    first_line = lines[0]
    if ';' in first_line:
        wsp_, prm_ = first_line.split(';')
        wsp_strs = [sym.strip() for sym in wsp_.split(',') if sym.strip()]
        wspolrzedne = [create_symbol(s) for s in wsp_strs]

        prm_ = prm_.strip()
        if prm_:
            par_strs = [sym.strip() for sym in prm_.split(',') if sym.strip()]
            parametry = [create_symbol(s) for s in par_strs]
    else:
        raise ValueError("First line must contain coordinates and parameters")

    # Parsuj komponenty metryki
    for line in lines[1:]:
        dat = line.split(maxsplit=2)
        if len(dat) == 3:
            try:
                i, j, expr = int(dat[0]), int(dat[1]), dat[2]
                symbols_dict = {str(sym): sym for sym in wspolrzedne + parametry}
                metryka[(i, j)] = sp.sympify(expr, locals=symbols_dict)
            except ValueError:
                raise ValueError(f"Invalid metric component: {line}")

    return wspolrzedne, parametry, metryka

    except Exception as e:
        raise ValueError(f"Metric parsing error: {str(e)}") 