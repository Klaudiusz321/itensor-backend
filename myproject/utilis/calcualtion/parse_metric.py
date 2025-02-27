from functools import lru_cache
import sympy as sp

def wczytaj_metryke_z_tekstu(metric_text: str):
    try:
        # Dodaj cache dla często używanych symboli
        @lru_cache(maxsize=128)
        def create_symbol(sym_name):
            sym_name = sym_name.strip()
            if not sym_name:
                raise ValueError("Empty symbol name")
                
            if sym_name in symbol_assumptions:
                return sp.Symbol(sym_name, **symbol_assumptions[sym_name])
            return sp.Symbol(sym_name)

        # Walidacja wejścia
        if len(metric_text) > 5000:
            raise ValueError("Metric text too long")

        # Szybsze parsowanie linii
        lines = [line.partition('#')[0].strip() for line in metric_text.splitlines()]
        lines = [line for line in lines if line]

        if not lines:
            raise ValueError("No valid lines in input")

        # Optymalizacja parsowania pierwszej linii
        first_line = lines[0]
        if ';' not in first_line:
            raise ValueError("First line must contain coordinates and parameters separated by ';'")

        wsp_, prm_ = first_line.split(';', 1)
        wspolrzedne = [create_symbol(s) for s in wsp_.split(',') if s.strip()]
        parametry = [create_symbol(s) for s in prm_.split(',') if s.strip()]

        # Pre-kompilacja wyrażeń symbolicznych
        symbols_dict = {str(sym): sym for sym in wspolrzedne + parametry}
        symbols_dict.update({
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'exp': sp.exp,
            'M': sp.Symbol('M', real=True, positive=True)
        })

        # Optymalizacja parsowania metryki
        n = len(wspolrzedne)
        metryka = {}
        
        for line in lines[1:]:
            try:
                if 'g_{' in line and '=' in line:
                    # Format g_{ij} = expr
                    idx_start = line.find('{') + 1
                    idx_end = line.find('}')
                    i, j = map(int, line[idx_start:idx_end].split(','))
                    expr = line.split('=')[1].strip()
                else:
                    # Format: i j expr
                    i, j, expr = line.split(maxsplit=2)
                    i, j = int(i), int(j)

                expr_sympy = sp.sympify(expr, locals=symbols_dict)
                metryka[(i, j)] = expr_sympy
                if i != j:
                    metryka[(j, i)] = expr_sympy
            except Exception as e:
                print(f"Warning: Skipping invalid line '{line}': {e}")
                continue

        return wspolrzedne, parametry, metryka

    except Exception as e:
        raise ValueError(f"Metric parsing error: {str(e)}") 