import sympy as sp
import re
import logging

logger = logging.getLogger(__name__)

def wczytaj_metryke_z_tekstu(metric_text: str):
    """
    Wczytuje metrykę z tekstu w formacie:
    x, y, z;
    0 0 -1
    1 1 r**2
    ...
    
    Zwraca:
    - listę współrzędnych
    - słownik z elementami metryki w formacie (i,j) -> wartość
    - słownik z oryginalnymi wyrażeniami tekstowymi
    """
    lines = metric_text.strip().split('\n')
    if not lines:
        logger.error("Pusty tekst metryki")
        raise ValueError("Pusty tekst metryki")
    
    # Wczytaj współrzędne z pierwszej linii
    coords_line = lines[0].strip()
    if ';' in coords_line:
        coords_text = coords_line.split(';')[0]
        wspolrzedne = [x.strip() for x in coords_text.split(',')]
        start_line = 1
    else:
        # Jeśli nie ma średnika, zakładamy domyślne współrzędne
        wspolrzedne = ['x', 'y', 'z', 't']
        start_line = 0
    
    # Liczba współrzędnych to wymiar przestrzeni
    n = len(wspolrzedne)
    logger.info(f"Wykryto {n} współrzędne: {', '.join(wspolrzedne)}")
    
    # Słownik z symbolami
    local_dict = {}
    
    # Zdefiniujmy podstawowe symbole
    for coord in wspolrzedne:
        local_dict[coord] = sp.Symbol(coord)
    
    # Dodajmy często używane symbole
    for sym in ['a', 'b', 'c', 'k', 'r', 'theta', 'phi', 'chi']:
        if sym not in local_dict:
            local_dict[sym] = sp.Symbol(sym)
    
    # Zdefiniujmy funkcję a(t) bezpośrednio
    t = local_dict['t']
    local_dict['a'] = sp.Function('a')(t)
    
    # Dodajmy funkcje trygonometryczne
    for func_name in ['sin', 'cos', 'tan', 'cot']:
        local_dict[func_name] = getattr(sp, func_name)
    
    # Przechowujemy oryginalne tekstowe reprezentacje wyrażeń metryki
    original_expressions = {}
    
    # Tworzenie słownika metryki
    metryka = {}
    for line_num, line in enumerate(lines[start_line:], start=start_line+1):
        line = line.strip()
        if not line:  # Pomijamy puste linie
            continue
            
        # Znajdujemy pierwsze dwa numery (indeksy) i resztę traktujemy jako wyrażenie
        parts = line.split(maxsplit=2)  # Dzielimy na maksymalnie 3 części
        if len(parts) != 3:
            raise ValueError(f"Błąd w linii {line_num}: Każda linia powinna zawierać indeks i, indeks j oraz wyrażenie g_ij")
            
        try:
            i, j = int(parts[0]), int(parts[1])
            if i >= n or j >= n or i < 0 or j < 0:
                raise ValueError(f"Błąd w linii {line_num}: Indeksy (i,j)=({i},{j}) poza zakresem [0,{n-1}]")
            
            # Zapisujemy oryginalny tekst wyrażenia
            original_expressions[(i, j)] = parts[2]
            original_expressions[(j, i)] = parts[2]  # symetria
            
            # Specjalna obsługa dla wyrażeń zawierających a(t)
            expr_str = parts[2]
            if "a(t)" in expr_str:
                # Zastąp a(t) przez a, które jest już zdefiniowane jako funkcja symboliczna
                expr_str = expr_str.replace("a(t)", "a")
            
            # Konwertujemy na wyrażenie SymPy do obliczeń
            try:
                value = sp.sympify(expr_str, locals=local_dict)
                metryka[(i,j)] = value
                metryka[(j,i)] = value  # metryka jest symetryczna
            except Exception as e:
                logger.error(f"Błąd parsowania wyrażenia '{expr_str}' w linii {line_num}: {str(e)}")
                raise ValueError(f"Nie można zinterpretować wyrażenia '{parts[2]}' w linii {line_num}")
            
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError(f"Błąd w linii {line_num}: Indeksy muszą być liczbami całkowitymi")
            raise
        except Exception as e:
            raise ValueError(f"Błąd przetwarzania metryki w linii: {line}") from e

    if not metryka:
        raise ValueError("Nie znaleziono żadnych komponentów metryki")

    # Sprawdzanie kompletności metryki
    for i in range(n):
        for j in range(n):
            if (i,j) not in metryka:
                metryka[(i,j)] = 0
                metryka[(j,i)] = 0
                original_expressions[(i, j)] = "0"
                original_expressions[(j, i)] = "0"

    logger.info(f"Metryka sparsowana pomyślnie. Wymiar: {n}")
    
    return wspolrzedne, metryka, original_expressions
