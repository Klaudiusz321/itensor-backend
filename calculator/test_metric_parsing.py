import re
import sympy as sp

def _fix_fn_expo(expr_str):
    """Fix function exponentiation with complex expressions.
    
    For example: sin(x)**2 -> (sin(x))**2
    or a(t)**2 -> (a(t))**2
    or (a + b*cos(theta))**2 -> ((a + b*cos(theta)))**2
    """
    # Handle function calls with arguments followed by exponentiation
    pattern1 = r'([a-zA-Z][a-zA-Z0-9_]*\([^()]+\))\*\*([0-9]+)'
    expr_str = re.sub(pattern1, r'(\1)**\2', expr_str)
    
    # Handle complex expressions with parentheses followed by exponentiation
    pattern2 = r'([a-zA-Z][a-zA-Z0-9_]*\([^()]+\))\*\*([a-zA-Z][a-zA-Z0-9_]*)'
    expr_str = re.sub(pattern2, r'(\1)**\2', expr_str)
    
    # Handle more complex expressions with nested parentheses
    pattern3 = r'(\([^()]+\))\*\*([0-9]+)'
    expr_str = re.sub(pattern3, r'\1**\2', expr_str)
    
    return expr_str

def test_fix_fn_expo():
    """Test the _fix_fn_expo function with different expressions."""
    test_cases = [
        # Simple function exponentials
        ('sin(x)**2', '(sin(x))**2'),
        ('a(t)**2', '(a(t))**2'),
        
        # Functions with complex arguments
        ('sin(x + y)**2', '(sin(x + y))**2'),
        
        # Variable exponentials
        ('a(t)**n', '(a(t))**n'),
        
        # Complex expressions with nested parentheses
        ('(a + b*cos(theta))**2', '(a + b*cos(theta))**2'),
        
        # Hyperbolic functions from de Sitter example
        ('a**2*cosh(tau)**2', 'a**2*(cosh(tau))**2'),
        ('a**2*cosh(tau)**2*sin(psi)**2', 'a**2*(cosh(tau))**2*(sin(psi))**2'),
        
        # More complex metric expressions
        ('a(t)**2 / (1 - k*psi**2)', '(a(t))**2 / (1 - k*psi**2)'),
        ('a(t)**2*r**2*sin(theta)**2', '(a(t))**2*r**2*(sin(theta))**2'),
    ]
    
    for input_expr, expected_output in test_cases:
        result = _fix_fn_expo(input_expr)
        print(f"Input: {input_expr}")
        print(f"Result: {result}")
        print(f"Expected: {expected_output}")
        print(f"Match: {result == expected_output}")
        print("-" * 50)

def test_function_parsing():
    """Test the function pattern matching and symbol creation."""
    # Sample metric string with various function patterns
    metric_str = """
    {
        "0,0": "-1",
        "1,1": "a(t)**2 / (1 - k*psi**2)",
        "2,2": "a(t)**2 * psi**2",
        "3,3": "a(t)**2 * psi**2 * sin(theta)**2",
        "4,4": "(a + b*cos(theta))**2"
    }
    """
    
    # Extract function patterns
    func_pattern = r'([a-zA-Z][a-zA-Z0-9_]*)\(([^()]+)\)'
    func_matches = re.findall(func_pattern, metric_str)
    
    # Process extracted functions
    symbols_dict = {}
    
    # First register all variable symbols that might be used
    all_possible_symbols = ['a', 'b', 'k', 'psi', 'theta', 't', 'r', 'tau', 'phi']
    for sym in all_possible_symbols:
        symbols_dict[sym] = sp.Symbol(sym)
    
    # Keep track of unique function-argument pairs
    func_arg_pairs = set()
    for func_name, arg_name in func_matches:
        func_arg_pairs.add((func_name, arg_name))
    
    print("Found function patterns:", func_matches)
    print("Unique function-argument pairs:", func_arg_pairs)
    
    # Define functions and arguments
    for func_name, arg_name in func_arg_pairs:
        # Remove any existing symbol with the function name to avoid conflicts
        if func_name in symbols_dict:
            del symbols_dict[func_name]
        
        # Create function
        symbols_dict[func_name] = sp.Function(func_name)
        
        print(f"Defined function {func_name}({arg_name})")
    
    # Add common math functions
    standard_funcs = ['sin', 'cos', 'sinh', 'cosh', 'tanh']
    for func_name in standard_funcs:
        if func_name not in symbols_dict:
            symbols_dict[func_name] = getattr(sp, func_name)
            print(f"Added standard function {func_name}")
    
    # Test by parsing a few expressions
    test_exprs = [
        "a(t)**2",
        "sin(theta)**2",
        "a(t)**2 * sin(psi)**2",
        "(a + b*cos(theta))**2"
    ]
    
    print("\nTesting expression parsing:")
    for expr_str in test_exprs:
        # Fix function exponentials
        expr_str_fixed = _fix_fn_expo(expr_str)
        
        try:
            # Parse the expression using sympy
            sympy_expr = sp.sympify(expr_str_fixed, locals=symbols_dict)
            print(f"Successfully parsed: {expr_str} -> {sympy_expr}")
        except Exception as e:
            print(f"Error parsing: {expr_str_fixed}, Error: {str(e)}")

if __name__ == "__main__":
    print("Testing _fix_fn_expo function:")
    print("=" * 50)
    test_fix_fn_expo()
    
    print("\nTesting function pattern matching:")
    print("=" * 50)
    test_function_parsing() 