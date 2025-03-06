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
    
    # Zdefiniujmy podstawowe symbole dla wszystkich współrzędnych
    for coord in wspolrzedne:
        local_dict[coord] = sp.Symbol(coord)
    
    # Dodajmy często używane symbole
    for sym in ['a', 'b', 'c', 'k', 'r', 'theta', 'phi', 'chi']:
        if sym not in local_dict:
            local_dict[sym] = sp.Symbol(sym)
    
    # Znajdź wszystkie wyrażenia typu a(t) w tekście metryki
    func_pattern = r'([a-zA-Z]+)\(([a-zA-Z]+)\)'
    func_matches = re.findall(func_pattern, metric_text)
    
    # Definiujemy funkcje znalezione w tekście
    for func_name, arg_name in func_matches:
        # Upewnij się, że argument funkcji jest zdefiniowany
        if arg_name not in local_dict:
            local_dict[arg_name] = sp.Symbol(arg_name)
        
        # Definiuj funkcję symboliczną
        arg_symbol = local_dict[arg_name]
        local_dict[func_name] = sp.Function(func_name)(arg_symbol)
        
        # Dodajemy również klucz dla całego wyrażenia funkcji (np. 'a(t)')
        func_key = f"{func_name}({arg_name})"
        local_dict[func_key] = local_dict[func_name]
    
    # Dodajmy funkcje trygonometryczne
    for func_name in ['sin', 'cos', 'tan', 'cot', 'exp', 'log']:
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
            
            # Konwertujemy na wyrażenie SymPy do obliczeń
            try:
                # Dla lepszego debugowania, pokazujemy wartości słownika
                if "a(t)" in parts[2]:
                    logger.info(f"Przetwarzam wyrażenie zawierające a(t): {parts[2]}")
                    if 'a(t)' in local_dict:
                        logger.info(f"a(t) jest zdefiniowane w słowniku jako: {local_dict['a(t)']}")
                    else:
                        logger.warning("a(t) nie jest zdefiniowane w słowniku!")
                
                # Konwersja wyrażenia na obiekt SymPy
                # Używamy trybu 'eval' dla lepszej obsługi funkcji
                value = sp.sympify(parts[2], locals=local_dict, evaluate=True)
                
                metryka[(i,j)] = value
                metryka[(j,i)] = value  # metryka jest symetryczna
                
            except Exception as e:
                logger.error(f"Błąd parsowania wyrażenia '{parts[2]}' w linii {line_num}: {str(e)}")
                
                # Próbujemy alternatywne podejście dla wyrażeń z funkcjami
                try:
                    # Zastępujemy a(t) bezpośrednio na a_t
                    modified_expr = re.sub(r'([a-zA-Z]+)\(([a-zA-Z]+)\)', r'\1_\2', parts[2])
                    logger.info(f"Próbuję zmodyfikowane wyrażenie: {modified_expr}")
                    
                    # Definiujemy nowe symbole dla funkcji
                    for func_name, arg_name in func_matches:
                        func_symbol_name = f"{func_name}_{arg_name}"
                        local_dict[func_symbol_name] = sp.Symbol(func_symbol_name)
                    
                    value = sp.sympify(modified_expr, locals=local_dict)
                    
                    metryka[(i,j)] = value
                    metryka[(j,i)] = value
                    
                except Exception as e2:
                    logger.error(f"Również alternatywne podejście nie działa: {str(e2)}")
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
