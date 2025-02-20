import sympy as sp

def generate_index_riemann(n):
    index = []
    for a in range(n):
        for b in range(a, n):
            for c in range(n):
                for d in range(c, n):
                    if (a * n + b) <= (c * n + d):
                        index.append((a, b, c, d))
    return index

def generate_index_ricci(n):
    index = []
    for i in range(n):
        for j in range(i, n):
            index.append((i, j))
    return index

def generate_index_christoffel(n):
    index = []
    for a in range(n):
        for b in range(n):
            for c in range(b, n):
                index.append((a, b, c))
    return index

def lower_indices(Riemann, g, n):
    R_abcd = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    R_abcd[a][b][c][d] = sum(g[a, i] * Riemann[i][b][c][d] for i in range(n))
    return R_abcd




def custom_simplify(expr):
    from sympy import simplify, factor, expand, trigsimp, cancel, ratsimp
    
    expr_simpl = expand(expr)
    expr_simpl = trigsimp(expr_simpl)
    expr_simpl = factor(expr_simpl)
    expr_simpl = simplify(expr_simpl)
    expr_simpl = cancel(expr_simpl)
    expr_simpl = ratsimp(expr_simpl)
    
    return expr_simpl
def process_latex(latex_str):
   
    def remove_function_argument(latex):
        result = ""
        i = 0
        while i < len(latex):
            if latex[i].isalpha():
                start = i
                while i < len(latex) and latex[i].isalpha():
                    i += 1
                func_name = latex[start:i]
                
                if i < len(latex) and latex[i] == '(':
                    i += 1  
                    arg_start = i
                    paren_count = 1
                    while i < len(latex) and paren_count > 0:
                        if latex[i] == '(':
                            paren_count += 1
                        elif latex[i] == ')':
                            paren_count -= 1
                        i += 1
                    arg = latex[arg_start:i-1].strip()
                    if arg in ['\\chi', 'chi']:
                       
                        result += func_name
                    else:
                        
                        result += f"{func_name}({arg})"
                else:
                   
                    result += func_name
            else:
                
                result += latex[i]
                i += 1
        return result

 
    def replace_derivative(latex):
        search_str = "\\frac{d}{d \\chi}"
        while search_str in latex:
            index = latex.find(search_str)
            
            after = latex[index + len(search_str):]
           
            var_end = index + len(search_str)
            while var_end < len(latex) and (latex[var_end].isalpha() or latex[var_end] == '\\'):
                var_end += 1
            var = latex[index + len(search_str):var_end].strip()
            
            if var.startswith('\\'):
                var_name = ""
                j = 0
                while j < len(var) and (var[j].isalpha() or var[j] == '\\'):
                    var_name += var[j]
                    j += 1
                var_replaced = f"{var_name}'"
            else:
                var_replaced = f"{var}'"
          
            latex = latex[:index] + var_replaced + latex[var_end:]
        return latex

    
    latex_str = remove_function_argument(latex_str)
   
    latex_str = replace_derivative(latex_str)

    
    return latex_str

def wczytaj_metryke_z_tekstu(metric_text: str):
    symbol_assumptions = {
        'a':    dict(real=True, positive=True),
        'tau':  dict(real=True),
        'psi':  dict(real=True),
        'theta':dict(real=True),
        'phi':  dict(real=True),
    }

    def create_symbol(sym_name):
        if sym_name in symbol_assumptions:
            return sp.Symbol(sym_name, **symbol_assumptions[sym_name])
        else:
            return sp.Symbol(sym_name)

    wspolrzedne = []
    parametry = []
    metryka = {}

    # Dzielimy tekst na linie
    lines = metric_text.splitlines()

    for line in lines:
        # Usuwamy komentarze i puste linie
        line = line.split('#')[0].strip()
        if not line:
            continue

        if ';' in line:
            wsp_, prm_ = line.split(';')
            wsp_strs = [sym.strip() for sym in wsp_.split(',') if sym.strip()]
            wspolrzedne = [create_symbol(s) for s in wsp_strs]

            prm_ = prm_.strip()
            if prm_:
                par_strs = [sym.strip() for sym in prm_.split(',') if sym.strip()]
                parametry = [create_symbol(s) for s in par_strs]
        else:
            dat = line.split(maxsplit=2)
            if len(dat) == 3:
                try:
                    i, j, expr = int(dat[0]), int(dat[1]), dat[2]
                    symbols_dict = {str(sym): sym for sym in wspolrzedne + parametry}
                    metryka[(i, j)] = sp.sympify(expr, locals=symbols_dict)
                except ValueError:
                    print(f"Error: Incorrect data in line: {line}")
    return wspolrzedne, parametry, metryka


def oblicz_tensory(wspolrzedne, metryka):
    n = len(wspolrzedne)

    g = sp.Matrix(n, n, lambda i, j: metryka.get((i, j), metryka.get((j, i), 0)))
    g_inv = g.inv()

    Gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for sigma in range(n):
        for mu in range(n):
            for nu in range(n):
                Gamma_sum = 0
                for lam in range(n):
                    partial_mu  = sp.diff(g[nu, lam], wspolrzedne[mu])
                    partial_nu  = sp.diff(g[mu, lam], wspolrzedne[nu])
                    partial_lam = sp.diff(g[mu, nu], wspolrzedne[lam])
                    Gamma_sum += g_inv[sigma, lam] * (partial_mu + partial_nu - partial_lam)
                Gamma[sigma][mu][nu] = custom_simplify(sp.Rational(1, 2) * Gamma_sum)

    Riemann = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for rho in range(n):
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    term1 = sp.diff(Gamma[rho][nu][sigma], wspolrzedne[mu])
                    term2 = sp.diff(Gamma[rho][mu][sigma], wspolrzedne[nu])
                    sum_term = 0
                    for lam in range(n):
                        sum_term += (Gamma[rho][mu][lam] * Gamma[lam][nu][sigma]
                                     - Gamma[rho][nu][lam] * Gamma[lam][mu][sigma])
                    Riemann[rho][sigma][mu][nu] = custom_simplify(term1 - term2 + sum_term)

    R_abcd = lower_indices(Riemann, g, n)

    Ricci = sp.zeros(n, n)
    for mu in range(n):
        for nu in range(n):
            Ricci[mu, nu] = custom_simplify(sum(Riemann[rho][mu][rho][nu] for rho in range(n)))
            Ricci[mu, nu] = custom_simplify(Ricci[mu, nu])

    Scalar_Curvature = custom_simplify(sum(g_inv[mu, nu] * Ricci[mu, nu] for mu in range(n) for nu in range(n)))
    Scalar_Curvature = custom_simplify(Scalar_Curvature)

    print("Final expression for lambdify:", Scalar_Curvature)

    return g, Gamma, R_abcd, Ricci, Scalar_Curvature

def compute_einstein_tensor(Ricci, Scalar_Curvature, g, g_inv, n):
    G_lower = sp.zeros(n, n)  
    G_upper = sp.zeros(n, n) 

    
    for mu in range(n):
        for nu in range(n):
            G_lower[mu, nu] = custom_simplify(Ricci[mu, nu] - sp.Rational(1, 2) * g[mu, nu] * Scalar_Curvature)


    for mu in range(n):
        for nu in range(n):
            sum_term = 0
            for alpha in range(n):
                sum_term += g_inv[mu, alpha] * G_lower[alpha, nu]
            G_upper[mu, nu] = custom_simplify(sum_term)

    return G_upper, G_lower

def generate_output(g, Gamma, R_abcd, Ricci, Scalar_Curvature, G_upper, G_lower, n):
    lines = []

    # METRYKA
    lines.append("Metric tensor components (textual format and LaTeX):")
    for i in range(n):
        for j in range(i, n):
            val = custom_simplify(g[i, j])
            if val != 0:
                lines.append(f"g_({i}{j}) = {val}")
                lines.append(f"g_{{{i}{j}}} = \\({sp.latex(val)}\\)")

    # SYMBOL CHRISTOFFELA
    lines.append("Non-zero Christoffel symbols (textual format and LaTeX):")
    for a in range(n):
        for b in range(n):
            for c in range(n):
                val = custom_simplify(Gamma[a][b][c])
                if val != 0:
                    lines.append(f"Γ^({a})_({b}{c}) = {val}")
                    lines.append(f"\\Gamma^{{{a}}}_{{{b}{c}}} = \\({sp.latex(val)}\\)")

    # TENSOR RIEMANNA
    lines.append("Non-zero components of the Riemann tensor:")
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    val = custom_simplify(R_abcd[a][b][c][d])
                    if val != 0:
                        lines.append(f"R_({a}{b}{c}{d}) = {val}")
                        lines.append(f"R_{{{a}{b}{c}{d}}} = \\({sp.latex(val)}\\)")

    # TENSOR RICCIEGO
    lines.append("Non-zero components of the Ricci tensor:")
    for i in range(n):
        for j in range(n):
            val = custom_simplify(Ricci[i, j])
            if val != 0:
                lines.append(f"R_({i}{j}) = {val}")
                lines.append(f"R_{{{i}{j}}} = \\({sp.latex(val)}\\)")

    # TENSOR EINSTEINA
    lines.append("Non-zero Einstein tensor components:")
    for i in range(n):
        for j in range(n):
            val = custom_simplify(G_lower[i, j])
            if val != 0:
                lines.append(f"G_({i}{j}) = {val}")
                lines.append(f"G_{{{i}{j}}} = \\({sp.latex(val)}\\)")

    # KRZYWIZNA SKALARNA
    if Scalar_Curvature != 0:
        lines.append("Scalar curvature R:")
        lines.append(f"R = {Scalar_Curvature}")
        lines.append(f"R = \\({sp.latex(Scalar_Curvature)}\\)")
        

    return "\n".join(lines)

def generate_christoffel_latex(Gamma, n):
    latex_symbols = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if not Gamma[i][j][k].equals(0):
                    latex = f"\\Gamma^{{{i}}}_{{{j}{k}}} = {sp.latex(Gamma[i][j][k])}"
                    latex_symbols.append(latex)
    return latex_symbols

def generate_riemann_latex(R_abcd, n):
    latex_symbols = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if not R_abcd[i][j][k][l].equals(0):
                        latex = f"R_{{{i}{j}{k}{l}}} = {sp.latex(R_abcd[i][j][k][l])}"
                        latex_symbols.append(latex)
    return latex_symbols

def generate_ricci_latex(Ricci, n):
    latex_symbols = []
    for i in range(n):
        for j in range(n):
            if not Ricci[i,j].equals(0):
                latex = f"R_{{{i}{j}}} = {sp.latex(Ricci[i,j])}"
                latex_symbols.append(latex)
    return latex_symbols

def generate_einstein_latex(G_lower, n):
    latex_symbols = []
    for i in range(n):
        for j in range(n):
            if not G_lower[i,j].equals(0):
                latex = f"G_{{{i}{j}}} = {sp.latex(G_lower[i,j])}"
                latex_symbols.append(latex)
    return latex_symbols



