import re
import sympy as sp

def _fix_fn_expo(expr_str):
    """Fix function exponentiation with complex expressions.
    
    For example: 
    - sin(x)**2 -> (sin(x))**2
    - a(t)**2 -> (a(t))**2
    - (a + b*cos(theta))**2 -> (a + b*cos(theta))**2
    - a**2*cosh(tau)**2 -> a**2*(cosh(tau))**2
    """
    # Apply regex substitutions in a specific order for proper handling
    
    # 1. Handle function calls with arguments followed by exponentiation with a number
    pattern1 = r'([a-zA-Z][a-zA-Z0-9_]*\([^()]+\))\*\*([0-9]+)'
    expr_str = re.sub(pattern1, r'(\1)**\2', expr_str)
    
    # 2. Handle function calls with arguments followed by exponentiation with a variable
    pattern2 = r'([a-zA-Z][a-zA-Z0-9_]*\([^()]+\))\*\*([a-zA-Z][a-zA-Z0-9_]*)'
    expr_str = re.sub(pattern2, r'(\1)**\2', expr_str)
    
    # 3. Handle trig/hyperbolic functions with exponentiation but without parentheses
    trig_funcs = ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh']
    for func in trig_funcs:
        pattern = fr'{func}\(([^()]+)\)\*\*([0-9]+|[a-zA-Z][a-zA-Z0-9_]*)'
        expr_str = re.sub(pattern, r'({0}(\1))**\2'.format(func), expr_str)
    
    # 4. Handle complex expressions in parentheses
    # We don't add extra parentheses here as they already exist
    pattern4 = r'(\([^()]+\))\*\*([0-9]+|[a-zA-Z][a-zA-Z0-9_]*)'
    expr_str = re.sub(pattern4, r'\1**\2', expr_str)
    
    return expr_str

def test_metric_parsing(metric_data, title):
    """Test parsing a specific metric structure."""
    print(f"\n=== Testing {title} ===")
    
    # Extract coordinates and parameters
    coords_params = metric_data.split('\n')[0].strip()
    coords, params = coords_params.split(';') if ';' in coords_params else (coords_params, '')
    
    coordinates = [x.strip() for x in coords.split(',')]
    parameters = [x.strip() for x in params.split(',')] if params.strip() else []
    
    print(f"Coordinates: {coordinates}")
    print(f"Parameters: {parameters}")
    
    # Create symbols dictionary
    symbols_dict = {}
    
    # Add coordinates as symbols
    for coord in coordinates:
        symbols_dict[coord] = sp.Symbol(coord)
    
    # Add parameters as symbols
    for param in parameters:
        if param not in symbols_dict:
            symbols_dict[param] = sp.Symbol(param)
    
    # Add common symbols that might be needed
    common_symbols = ['k', 'r']
    for sym in common_symbols:
        if sym not in symbols_dict:
            symbols_dict[sym] = sp.Symbol(sym)
    
    # Add standard functions
    standard_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sinh', 'cosh', 'tanh', 'sqrt']
    for func_name in standard_funcs:
        symbols_dict[func_name] = getattr(sp, func_name)
    
    # Extract and process metric components
    metric_dict = {}
    metric_lines = [line.strip() for line in metric_data.split('\n')[1:] if line.strip()]
    
    for line in metric_lines:
        parts = line.split(maxsplit=2)
        if len(parts) == 3:
            i, j, expr_str = int(parts[0]), int(parts[1]), parts[2]
            
            # Process function patterns
            func_pattern = r'([a-zA-Z][a-zA-Z0-9_]*)\(([^()]+)\)'
            func_matches = re.findall(func_pattern, expr_str)
            
            # Create functions
            for func_name, arg_name in func_matches:
                if func_name in symbols_dict:
                    # If it's already a symbol, convert to a function
                    del symbols_dict[func_name]
                
                symbols_dict[func_name] = sp.Function(func_name)
                print(f"Defined function {func_name}({arg_name})")
            
            # Apply _fix_fn_expo to handle function exponentiation
            expr_str_fixed = _fix_fn_expo(expr_str)
            print(f"Fixed expression: {expr_str} -> {expr_str_fixed}")
            
            try:
                # Try to parse the expression
                sympy_expr = sp.sympify(expr_str_fixed, locals=symbols_dict)
                metric_dict[(i, j)] = sympy_expr
                print(f"Successfully parsed metric component [{i}][{j}]: {sympy_expr}")
            except Exception as e:
                print(f"Error parsing metric component [{i}][{j}]: {expr_str_fixed}, Error: {str(e)}")
    
    return metric_dict

# Test metrics from the images
metric_torus = """theta, phi; a, b
0 0 b**2
1 1 (a + b*cos(theta))**2"""

de_sitter = """tau, psi, theta, phi; a
0 0 -a**2
1 1 a**2*cosh(tau)**2
2 2 a**2*cosh(tau)**2*sin(psi)**2
3 3 a**2*cosh(tau)**2*sin(psi)**2*sin(theta)**2"""

# Add FLRW metric example
flrw_metric = """t, r, theta, phi; a, k
0 0 -1
1 1 a(t)**2 / (1 - k*r**2)
2 2 a(t)**2 * r**2
3 3 a(t)**2 * r**2 * sin(theta)**2"""

# Add Schwarzschild metric example
schwarzschild_metric = """t, r, theta, phi; c, G, M
0 0 -c**2*(1 - 2*G*M/(r*c**2))
1 1 1/(1 - 2*G*M/(r*c**2))
2 2 r**2
3 3 r**2*sin(theta)**2"""

if __name__ == "__main__":
    print("Testing example metrics from the images...")
    
    # Test Metric Torus
    metric_dict1 = test_metric_parsing(metric_torus, "Metric Torus in Euclidean Space")
    
    # Test De Sitter Spacetime
    metric_dict2 = test_metric_parsing(de_sitter, "Four-Dimensional de Sitter Spacetime")
    
    # Test FLRW metric
    metric_dict3 = test_metric_parsing(flrw_metric, "FLRW Metric")
    
    # Test Schwarzschild metric
    metric_dict4 = test_metric_parsing(schwarzschild_metric, "Schwarzschild Metric") 