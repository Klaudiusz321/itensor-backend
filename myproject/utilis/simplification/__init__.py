from .expand import expand_expr
from .trig_simplify import trig_simplify
from .log_combine import log_combine
from .pow_simplify import pow_simplify
from .together import together_expr
from .algebraic_simplify import algebraic_simplify
from .trig_conversion import trig_conversion
from .powdenest import powdenest_expr
from .sqrtdenest import sqrtdenest_expr

def custom_simplify(expr, max_iter=5):
    """
    Wykonuje iteracyjne uproszczenie wyrażenia.
    """
    if not hasattr(expr, 'equals'):
        raise TypeError("Input must be a sympy expression (sp.Basic).")
    
    iteration = 0
    prev_expr = expr
    while iteration < max_iter:
        new_expr = expand_expr(prev_expr, deep=True)
        new_expr = trig_simplify(new_expr, method='fu')
        new_expr = log_combine(new_expr, force=True)
        new_expr = pow_simplify(new_expr, deep=True)
        
        new_expr = algebraic_simplify(new_expr, rational=True)
        new_expr = trig_simplify(new_expr, method='fu')
        new_expr = trig_conversion(new_expr)
        new_expr = powdenest_expr(new_expr, force=True)
        new_expr = sqrtdenest_expr(new_expr)
        new_expr = trig_simplify(new_expr, method='fu')
        new_expr = together_expr(new_expr)
        
        if new_expr.equals(prev_expr):
            break
        prev_expr = new_expr
        iteration += 1

    return new_expr

# Eksportujemy wszystkie funkcje
__all__ = [
    'custom_simplify',
    'expand_expr',
    'trig_simplify',
    'log_combine',
    'pow_simplify',
    'together_expr',
    'algebraic_simplify',
    'trig_conversion',
    'powdenest_expr',
    'sqrtdenest_expr'
]
