import sympy as sp
import logging
import re
from fractions import Fraction

logger = logging.getLogger(__name__)

def custom_simplify(expr, max_depth=3, current_depth=0):
    """
    Custom simplification function for SymPy expressions.
    Uses multiple simplification strategies.
    
    Args:
        expr: SymPy expression to simplify
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
        
    Returns:
        Simplified SymPy expression
    """
    if current_depth > max_depth:
        logger.debug(f"Reached max simplification depth {max_depth}")
        return expr
    
    try:
        # Apply various simplifications in a specific order
        logger.debug(f"Before simplification: {expr}")
        
        # Skip for very complex expressions to avoid long computation times
        if expr.count_ops() > 500:
            logger.warning(f"Expression too complex ({expr.count_ops()} operations), limiting simplification")
            expr = sp.powsimp(expr)
            expr = sp.trigsimp(expr)
            return expr
        
        # Convert floats to fractions for exact arithmetic
        expr = convert_to_fractions(expr)
        
        # Basic simplifications first
        expr = sp.cancel(expr)
        
        # For expressions with trigonometric functions, apply trigsimp
        if any(func in str(expr) for func in ('sin', 'cos', 'tan', 'cot', 'sec', 'csc')):
            expr = sp.trigsimp(expr)
            expr = sp.simplify(expr)
        
        # For expressions with hyperbolic functions, apply hypersimp
        if any(func in str(expr) for func in ('sinh', 'cosh', 'tanh')):
            expr = sp.trigsimp(expr)
            expr = sp.simplify(expr)
        
        # For expressions with square roots, apply root simplification
        if any(func in str(expr) for func in ('sqrt', 'root')):
            expr = sp.radsimp(expr)
            expr = sp.simplify(expr)
        
        # For expressions with complex exponents
        expr = sp.powsimp(expr)
        
        # General simplification
        expr = sp.simplify(expr)
        
        # Handle rational functions
        expr = sp.ratsimp(expr)
        
        # For expressions with derivatives
        if any(str(expr).find(func) != -1 for func in ('Derivative', 'diff')):
            try:
                # Try to apply special rules for derivatives
                expr = sp.simplify(expr)
                # Collect terms with similar derivatives
                expr = sp.collect(expr, sp.Symbol('a'))
            except Exception as e:
                logger.warning(f"Error in derivative simplification: {e}")
        
        # Add special handling for FLRW metric expressions with a(t) derivatives
        if 'a(' in str(expr) and ('Derivative' in str(expr) or 'diff' in str(expr)):
            try:
                # Normalize notation for a(t), a'(t), a''(t)
                # This helps match the expected notation for FLRW scalar curvature
                if 'Derivative' in str(expr):
                    t_symbol = None
                    a_func = None
                    
                    # Try to extract symbols
                    for sym in expr.free_symbols:
                        if str(sym) == 't':
                            t_symbol = sym
                            break
                    
                    # Look for a(t) function
                    for atom in expr.atoms(sp.Function):
                        if str(atom.func) == 'a':
                            a_func = atom
                            break
                    
                    if t_symbol and a_func:
                        # Replace Derivative(a(t), t) with a_dot
                        a_dot_expr = sp.Symbol('a_dot')
                        a_ddot_expr = sp.Symbol('a_ddot')
                        
                        # Create substitution maps
                        # Replace Derivative(a(t), t) with a_dot
                        # Replace Derivative(a(t), (t, 2)) with a_ddot
                        repl_map = {}
                        
                        # Find all derivatives in the expression
                        for atom in expr.atoms(sp.Derivative):
                            if atom.expr == a_func and atom.variables == (t_symbol,):
                                repl_map[atom] = a_dot_expr
                            elif atom.expr == a_func and atom.variables == (t_symbol, t_symbol):
                                repl_map[atom] = a_ddot_expr
                        
                        if repl_map:
                            expr = expr.subs(repl_map)
                            # Rearrange to standard form for FLRW scalar curvature
                            expr = sp.collect(expr, [a_dot_expr, a_ddot_expr])
            except Exception as e:
                logger.warning(f"Error in FLRW simplification: {e}")
        
        # Try one more simplification round with the combined result
        expr = sp.simplify(expr)
        
        logger.debug(f"After simplification: {expr}")
        return expr
        
    except Exception as e:
        logger.warning(f"Error during simplification: {e}")
        return expr

def replace_inverse_trig_in_string(expr_string):
    """
    Replace inverse trigonometric functions in a string to improve parsing.
    
    Example: arcsin(x) -> asin(x)
    """
    replacements = [
        (r'arcsin\(', 'asin('),
        (r'arccos\(', 'acos('),
        (r'arctan\(', 'atan('),
        (r'arccot\(', 'acot('),
        (r'arcsec\(', 'asec('),
        (r'arccsc\(', 'acsc('),
    ]
    
    for pattern, replacement in replacements:
        expr_string = re.sub(pattern, replacement, expr_string)
    
    return expr_string

def weyl_simplify(expr):
    """
    Specialized simplification for Weyl tensor components.
    Weyl tensor components often have more complex structure.
    """
    if expr == 0:
        return expr
        
    try:
        # Skip very complex expressions
        if expr.count_ops() > 1000:
            logger.warning(f"Weyl component too complex ({expr.count_ops()} operations), limiting simplification")
            return expr
            
        # First apply fraction conversion
        expr = convert_to_fractions(expr)
        
        # Basic algebraic simplification
        expr = sp.powsimp(expr)
        expr = sp.cancel(expr)
        
        # Try to combine similar terms
        expr = sp.collect(expr, expr.free_symbols)
        
        # For expressions with trig functions, use trigsimp
        if any(func in str(expr) for func in ('sin', 'cos', 'tan')):
            expr = sp.trigsimp(expr)
        
        # Factor common terms
        expr = sp.factor(expr)
        
        return expr
    except Exception as e:
        logger.warning(f"Error in Weyl component simplification: {e}")
        return expr

def replace_floats_in_string(expr_str):
    """
    Zastępuje liczby zmiennoprzecinkowe w ciągu znaków ich odpowiednikami w postaci ułamków.
    """
    # Wzorzec regex do wyszukiwania liczb zmiennoprzecinkowych
    float_pattern = r'[-+]?\d+\.\d+'
    
    def replace_match(match):
        float_str = match.group(0)
        try:
            float_val = float(float_str)
            frac = Fraction(float_val).limit_denominator(100)
            if frac.denominator == 1:
                return str(frac.numerator)
            else:
                return f"{frac.numerator}/{frac.denominator}"
        except ValueError:
            return float_str
    
    return re.sub(float_pattern, replace_match, expr_str)

def replace_greek_letters(expr_str):
    """
    Zamienia nazwy greckich liter na ich odpowiedniki w LaTeX.
    """
    greek_letters = {
        'alpha': 'alpha',
        'beta': 'beta',
        'gamma': 'gamma',
        'delta': 'delta',
        'epsilon': 'epsilon',
        'zeta': 'zeta',
        'eta': 'eta',
        'theta': 'theta',
        'iota': 'iota',
        'kappa': 'kappa',
        'lambda': 'lambda',
        'mu': 'mu',
        'nu': 'nu',
        'xi': 'xi',
        'omicron': 'omicron',
        'pi': 'pi',
        'rho': 'rho',
        'sigma': 'sigma',
        'tau': 'tau',
        'upsilon': 'upsilon',
        'phi': 'phi',
        'chi': 'chi',
        'psi': 'psi',
        'omega': 'omega'
    }
    
    # Dopasuj tylko całe słowa
    for greek, latex in greek_letters.items():
        expr_str = re.sub(r'\b' + greek + r'\b', f"\\{latex}", expr_str)
    
    return expr_str

def convert_to_fractions(expr):
    """
    Convert floating-point numbers in an expression to fractions.
    """
    try:
        if expr.is_Number and float(expr).is_integer():
            return sp.Integer(int(float(expr)))
        elif expr.is_Number and not expr.is_Integer:
            try:
                value = float(expr)
                frac = Fraction(value).limit_denominator(1000)
                return sp.Rational(frac.numerator, frac.denominator)
            except (ValueError, OverflowError, TypeError):
                return expr
        elif expr.is_Add:
            return sp.Add(*[convert_to_fractions(arg) for arg in expr.args])
        elif expr.is_Mul:
            return sp.Mul(*[convert_to_fractions(arg) for arg in expr.args])
        elif expr.is_Pow:
            base, exp = expr.as_base_exp()
            return sp.Pow(convert_to_fractions(base), convert_to_fractions(exp))
        elif expr.is_Function:
            return expr.func(*[convert_to_fractions(arg) for arg in expr.args])
        else:
            return expr
    except (AttributeError, ValueError, TypeError) as e:
        logger.warning(f"Error in convert_to_fractions: {e}, returning original expression")
        return expr
