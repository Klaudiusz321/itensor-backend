import sympy as sp
import logging
import re
from fractions import Fraction

logger = logging.getLogger(__name__)

def custom_simplify(expr):
    """
    Apply custom simplification rules for tensor expressions using the sequence
    from the original code that worked correctly.
    """
    if expr is None:
        return None
    
    try:
        # Use the exact sequence from the original code that produced correct results
        expr_simpl = sp.expand(expr)
        expr_simpl = sp.trigsimp(expr_simpl)
        expr_simpl = sp.factor(expr_simpl)
        expr_simpl = sp.simplify(expr_simpl)
        expr_simpl = sp.cancel(expr_simpl)
        expr_simpl = sp.ratsimp(expr_simpl)
        
        return expr_simpl
    except Exception as e:
        logger.error(f"Error in custom simplification: {e}")
        return expr  # Return original expression if simplification fails

def weyl_simplify(expr):
    """Special simplified for Weyl tensor components which often have complex patterns"""
    try:
        if expr is None or expr == 0:
            return expr
            
        # Use the same sequence as custom_simplify for consistency
        expr_simpl = sp.expand(expr)
        expr_simpl = sp.trigsimp(expr_simpl)
        expr_simpl = sp.factor(expr_simpl)
        expr_simpl = sp.simplify(expr_simpl)
        expr_simpl = sp.cancel(expr_simpl)
        expr_simpl = sp.ratsimp(expr_simpl)
        
        return expr_simpl
    except Exception as e:
        logger.error(f"Error in Weyl tensor simplification: {e}")
        return expr

def replace_inverse_trig_in_string(expr_str):
    """Replace arcsin, arccos, etc. with asin, acos, etc. for consistency"""
    replacements = [
        (r'arcsin', 'asin'),
        (r'arccos', 'acos'), 
        (r'arctan', 'atan'),
        (r'arccot', 'acot'),
        (r'arccsc', 'acsc'),
        (r'arcsec', 'asec')
    ]
    
    for old, new in replacements:
        expr_str = re.sub(old, new, expr_str)
        
    return expr_str

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
    Convert decimal numbers in an expression to fractions.
    """
    if expr is None:
        return "0"
        
    # Convert to string first
    expr_str = str(expr)
    
    # Special handling for constant curvature spaces
    
    # 3D sphere with parameter a
    if expr_str == "6/a**2" or (isinstance(expr, sp.Expr) and 
                                sp.simplify(expr - 6/sp.Symbol('a')**2) == 0):
        return "6/a**2"
    
    # de Sitter space
    if expr_str == "12/a**2" or (isinstance(expr, sp.Expr) and 
                                sp.simplify(expr - 12/sp.Symbol('a')**2) == 0):
        return "12/a**2"
    
    # Special handling for common patterns in spherical metrics
    if 'cos(psi)/sin(psi)' in expr_str:
        expr_str = expr_str.replace('cos(psi)/sin(psi)', '1/tan(psi)')
    if 'cos(theta)/sin(theta)' in expr_str:
        expr_str = expr_str.replace('cos(theta)/sin(theta)', '1/tan(theta)')
    
    # Replace decimals with fractions
    float_pattern = r'[-+]?[0-9]*\.[0-9]+'
    
    def replace_with_fraction(match):
        decimal_str = match.group(0)
        try:
            decimal = float(decimal_str)
            fraction = Fraction(decimal).limit_denominator(100)
            if fraction.denominator == 1:
                return str(fraction.numerator)
            else:
                return f"{fraction.numerator}/{fraction.denominator}"
        except ValueError:
            return decimal_str
    
    # Apply the replacement
    result = re.sub(float_pattern, replace_with_fraction, expr_str)
    
    # Replace inverse trigonometric functions for consistency
    result = replace_inverse_trig_in_string(result)
    
    # Special case for sin(2*theta) and similar forms
    if 'sin(2*theta)' in result:
        result = result.replace('2*sin(theta)*cos(theta)', 'sin(2*theta)')
    if 'sin(2*psi)' in result:
        result = result.replace('2*sin(psi)*cos(psi)', 'sin(2*psi)')
    
    return result
