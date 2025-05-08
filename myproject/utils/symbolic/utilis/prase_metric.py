import sympy as sp
import re
import logging

logger = logging.getLogger(__name__)


class PraseMetric:
    def __init__(self, metric_text: str):
        self.metric_text = metric_text

    

    def wczytaj_metryke_z_tekstu(self):
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
        lines = self.metric_text.strip().split('\n')
        if not lines:
            logger.error("Pusty tekst metryki")
            raise ValueError("Pusty tekst metryki")
        
        # Check if this might be an FLRW-style metric format
        if len(lines) > 1 and any(";" in line for line in lines[:2]) and any(re.search(r'\bk\b', line) for line in lines[:2]):
            try:
                return self.parse_flrw_metric(self.metric_text)
            except Exception as e:
                logger.warning(f"Failed to parse as FLRW metric, falling back to standard format: {str(e)}")
                # Continue with standard parsing
        
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
        for sym in ['b', 'c', 'k', 'r', 'theta', 'phi', 'chi']:
            if sym not in local_dict:
                local_dict[sym] = sp.Symbol(sym)
        
        # Znajdź wszystkie wyrażenia typu a(t) w tekście metryki
        func_pattern = r'([a-zA-Z]+)\(([a-zA-Z]+)\)'
        func_matches = re.findall(func_pattern, self.metric_text)
        
        # Also check for parameter 'a' which is common in de Sitter metrics
        if ';' in coords_line and 'a' in coords_line.split(';')[1]:
            logger.info("Detected parameter 'a' in coordinate line")
            local_dict['a'] = sp.Symbol('a')
        
        # Definiujemy funkcje znalezione w tekście
        for func_name, arg_name in func_matches:
            # Upewnij się, że argument funkcji jest zdefiniowany
            if arg_name not in local_dict:
                local_dict[arg_name] = sp.Symbol(arg_name)
            
            # Important: Delete any existing symbol with the function name to avoid conflicts
            if func_name in local_dict:
                del local_dict[func_name]
            
            # Get the argument symbol
            arg_symbol = local_dict[arg_name]
            
            # Create a SymPy Function
            func = sp.Function(func_name)
            local_dict[func_name] = func
            
            # Add the function applied to the argument
            func_expr = func(arg_symbol)
            
            # Add a key for the entire function expression (e.g., 'a(t)')
            func_key = f"{func_name}({arg_name})"
            local_dict[func_key] = func_expr
        
        # Dodajmy funkcje trygonometryczne i hiperboliczne
        for func_name in ['sin', 'cos', 'tan', 'cot', 'exp', 'log', 'sinh', 'cosh', 'tanh']:
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

    def parse_flrw_metric(metric_text: str):
        """
        Parses FLRW-style metric input format like:
        
        t, psi, theta, phi; k
        0 0 -c**2
        1 1 a(t)**2 / (1 - k*psi**2)
        2 2 a(t)**2 * psi**2
        3 3 a(t)**2 * psi**2 * sin(theta)**2
        
        Returns:
        - list of coordinates
        - dictionary with metric elements in format (i,j) -> value
        - dictionary with original expressions
        """
        lines = metric_text.strip().split('\n')
        if not lines:
            logger.error("Empty metric text")
            raise ValueError("Empty metric text")
        
        # Get coordinates and parameters from the first line
        coords_line = lines[0].strip()
        if ';' not in coords_line:
            raise ValueError("FLRW format requires coordinates and parameters separated by ';' in the first line")
        
        coords_text, params_text = coords_line.split(';', 1)
        coordinates = [x.strip() for x in coords_text.split(',')]
        parameters = [x.strip() for x in params_text.split(',') if x.strip()]
        
        n = len(coordinates)
        logger.info(f"Detected {n} coordinates: {', '.join(coordinates)} and parameters: {', '.join(parameters)}")
        
        # Set up the symbol dictionary
        local_dict = {}
        
        # Define symbols for all coordinates
        for coord in coordinates:
            local_dict[coord] = sp.Symbol(coord)
        
        # Define symbols for all parameters
        for param in parameters:
            local_dict[param] = sp.Symbol(param)
        
        # Add commonly used symbols if not already defined
        common_symbols = ['b', 'c', 'k', 'r', 'theta', 'phi', 'chi']
        for sym in common_symbols:
            if sym not in local_dict:
                local_dict[sym] = sp.Symbol(sym)
        
        # Find all expressions like a(t) in the metric text
        func_pattern = r'([a-zA-Z]+)\(([a-zA-Z]+)\)'
        func_matches = re.findall(func_pattern, metric_text)
        
        # First process all function matches to create proper SymPy Function objects
        for func_name, arg_name in func_matches:
            # Ensure the function argument is defined
            if arg_name not in local_dict:
                local_dict[arg_name] = sp.Symbol(arg_name)
            
            # Important: Remove any existing symbol with this name to avoid conflicts
            if func_name in local_dict:
                del local_dict[func_name]
            
            # Get the argument symbol
            arg_symbol = local_dict[arg_name]
            
            # Create a SymPy Function
            func = sp.Function(func_name)
            local_dict[func_name] = func
            
            # The function applied to the argument 
            func_expr = func(arg_symbol)
            
            # Add a key for the entire function expression (e.g., 'a(t)')
            func_key = f"{func_name}({arg_name})"
            local_dict[func_key] = func_expr
            
            logger.info(f"Defined function {func_name}({arg_name}) = {func_expr}")
        
        # Add common trigonometric, hyperbolic, and mathematical functions
        for func_name in ['sin', 'cos', 'tan', 'cot', 'exp', 'log', 'sinh', 'cosh', 'tanh']:
            local_dict[func_name] = getattr(sp, func_name)
        
        # Store original text expressions
        original_expressions = {}
        
        # Create the metric dictionary
        metric = {}
        for line_num, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Find indices and expression
            parts = line.split(maxsplit=2)
            if len(parts) != 3:
                raise ValueError(f"Error in line {line_num}: Each line should contain index i, index j, and expression g_ij")
            
            try:
                i, j = int(parts[0]), int(parts[1])
                if i >= n or j >= n or i < 0 or j < 0:
                    raise ValueError(f"Error in line {line_num}: Indices (i,j)=({i},{j}) out of range [0,{n-1}]")
                
                # Save the original expression text
                expr_text = parts[2]
                original_expressions[(i, j)] = expr_text
                original_expressions[(j, i)] = expr_text  # symmetry
                
                # Fix common notation differences (^ for power instead of **)
                expr_text = expr_text.replace('^', '**')
                
                # Log the expression we're about to parse
                logger.info(f"Parsing expression: {expr_text} with dictionary keys: {list(local_dict.keys())}")
                
                # Convert to SymPy expression
                try:
                    value = sp.sympify(expr_text, locals=local_dict, evaluate=True)
                    metric[(i, j)] = value
                    metric[(j, i)] = value  # metric is symmetric
                    logger.info(f"Successfully parsed expression {expr_text} to {value}")
                except Exception as e:
                    logger.error(f"Error parsing expression '{expr_text}' in line {line_num}: {str(e)}")
                    
                    # Try alternative approach for expressions with functions
                    try:
                        # Replace a(t) directly with a_t
                        modified_expr = re.sub(r'([a-zA-Z]+)\(([a-zA-Z]+)\)', r'\1_\2', expr_text)
                        logger.info(f"Trying modified expression: {modified_expr}")
                        
                        # Define new symbols for functions
                        for func_name, arg_name in func_matches:
                            func_symbol_name = f"{func_name}_{arg_name}"
                            local_dict[func_symbol_name] = sp.Symbol(func_symbol_name)
                        
                        value = sp.sympify(modified_expr, locals=local_dict)
                        metric[(i, j)] = value
                        metric[(j, i)] = value
                        logger.info(f"Successfully parsed with modified expression {modified_expr} to {value}")
                    except Exception as e2:
                        logger.error(f"Alternative approach also fails: {str(e2)}")
                        raise ValueError(f"Cannot interpret expression '{expr_text}' in line {line_num}")
                
            except ValueError as e:
                if "invalid literal for int()" in str(e):
                    raise ValueError(f"Error in line {line_num}: Indices must be integers")
                raise
            except Exception as e:
                raise ValueError(f"Error processing metric in line: {line}") from e
        
        if not metric:
            raise ValueError("No metric components found")
        
        # Check for completeness of the metric
        for i in range(n):
            for j in range(n):
                if (i, j) not in metric:
                    metric[(i, j)] = 0
                    metric[(j, i)] = 0
                    original_expressions[(i, j)] = "0"
                    original_expressions[(j, i)] = "0"
        
        logger.info(f"FLRW metric parsed successfully. Dimension: {n}")
        return coordinates, metric, original_expressions
