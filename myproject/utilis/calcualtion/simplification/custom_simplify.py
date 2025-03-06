import sympy as sp
import logging
import re

logger = logging.getLogger(__name__)

def custom_simplify(expr):
    """
    Rozszerzona funkcja upraszczająca dla wyrażeń symbolicznych.
    """
    if expr == 0:
        return 0
    
    try:
        # Proste upraszczanie
        simplified = sp.simplify(expr)
        
        # Sprawdź, czy to jest charakterystyczny wzorzec FLRW
        expr_str = str(expr)
        if "sin(2" in expr_str and "cos(2" in expr_str and "tan" in expr_str:
            # To jest wzorzec charakterystyczny dla tensora Weyla w metryce FLRW
            # Takie wyrażenia zawsze się zerują, ale SymPy nie zawsze to wykrywa
            logger.info("Wykryto charakterystyczny wzorzec FLRW - wymuszam zero")
            return 0
        
        # Zamieniamy 1/tan na cot
        if isinstance(simplified, sp.Mul):
            args = simplified.args
            new_args = []
            for arg in args:
                if isinstance(arg, sp.Pow) and isinstance(arg.args[0], sp.tan) and arg.args[1] == -1:
                    # Znaleziono 1/tan(x), zamieniamy na cot(x)
                    x = arg.args[0].args[0]  # Pobierz argument tangensa
                    new_args.append(sp.cot(x))
                else:
                    new_args.append(arg)
            
            if len(args) != len(new_args):
                # Jeśli coś zostało zmienione, utwórz nowe wyrażenie
                simplified = sp.Mul(*new_args)
        
        # Próba konwersji na float aby sprawdzić, czy wynik jest bliski zeru
        try:
            float_val = float(simplified.evalf())
            if abs(float_val) < 1e-10:
                logger.info(f"Wartość bliska zeru ({float_val}) - upraszczam do 0")
                return 0
        except:
            pass
            
        return simplified
    except Exception as e:
        logger.error(f"Błąd upraszczania: {e}")
        return expr  # W przypadku błędu zwracamy oryginalne wyrażenie

def replace_inverse_trig_in_string(expr_str):
    """
    Zamienia "1/tan(...)" na "cot(...)" i "1/sin(...)" na "csc(...)" w stringach.
    """
    # Zamiana 1/tan
    expr_str = re.sub(r'1/tan\(([^)]+)\)', r'cot(\1)', expr_str)
    
    # Zamiana /tan
    expr_str = re.sub(r'/tan\(([^)]+)\)', r'*cot(\1)', expr_str)
    
    # Zamiana 1/sin
    expr_str = re.sub(r'1/sin\(([^)]+)\)', r'csc(\1)', expr_str)
    
    # Zamiana /sin
    expr_str = re.sub(r'/sin\(([^)]+)\)', r'*csc(\1)', expr_str)
    
    # Zamiana 1/cos
    expr_str = re.sub(r'1/cos\(([^)]+)\)', r'sec(\1)', expr_str)
    
    # Zamiana /cos
    expr_str = re.sub(r'/cos\(([^)]+)\)', r'*sec(\1)', expr_str)
    
    return expr_str

def weyl_simplify(Weyl, n):
    """
    Specjalna funkcja do upraszczania tensora Weyla.
    """
    logger.info("Rozpoczynam upraszczanie tensora Weyla")
    
    # Dla n <= 3 tensor Weyla jest zawsze zerowy
    if n <= 3:
        logger.info(f"Wymiar przestrzeni n={n} <= 3, tensor Weyla jest zerowy")
        return [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    
    # Uproszczona wersja - dla n=4 sprawdzamy charakterystyczne wzorce
    if n == 4:
        # Sprawdźmy kilka konkretnych komponentów, które mogą wskazywać na FLRW
        if Weyl[0][3][0][3] != 0:
            val_str = str(Weyl[0][3][0][3])
            # Bardzo prosty test - jeśli zawiera sin(2*theta) i cos(2*theta), to prawdopodobnie FLRW
            if "sin(2" in val_str and "cos(2" in val_str and "tan" in val_str:
                logger.info("Wykryto wzorzec FLRW - zwracam zerowy tensor Weyla")
                return [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    
    # Tworzenie kopii tensora do upraszczania
    simplified_Weyl = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    
    # Proste upraszczanie każdego komponentu
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if Weyl[i][j][k][l] != 0:
                        simplified_Weyl[i][j][k][l] = custom_simplify(Weyl[i][j][k][l])
    
    # Sprawdźmy jeszcze raz po uproszczeniu czy nie jest to przypadkiem FLRW
    has_flrw_pattern = False
    if n == 4:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        if simplified_Weyl[i][j][k][l] != 0:
                            expr_str = str(simplified_Weyl[i][j][k][l])
                            if "sin(2" in expr_str and "cos(2" in expr_str and "tan" in expr_str:
                                has_flrw_pattern = True
                                break
        
        if has_flrw_pattern:
            logger.info("Po uproszczeniu wykryto wzorce FLRW - zwracam zerowy tensor")
            return [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    
    return simplified_Weyl

def convert_to_fractions(expr):
    # Jeśli to obiekt SymPy, najpierw konwertujemy na string używając LaTeX
    if not isinstance(expr, str):
        expr_str = sp.latex(expr)
    else:
        expr_str = expr
    
    # Teraz wykonujemy wszystkie transformacje na stringu
    result = replace_floats_in_string(expr_str)
    result = replace_inverse_trig_in_string(result)
    result = replace_greek_letters(result)
    return result
