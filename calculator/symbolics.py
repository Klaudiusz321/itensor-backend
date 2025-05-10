# calculator/symbolics.py
import logging
from myproject.utils.symbolic.compute_tensor import ComputeTensor

logger = logging.getLogger(__name__)

def compute_symbolic(*, dimension, coords, metric, evaluation_point=None):
    """
    Entry-point for symbolic tensor calculations.
    
    Args:
        dimension (int): The dimension of the space
        coords (list[str]): List of coordinate names
        metric (list[list[str]]): The metric tensor as a 2D array of strings or sympy expressions
        evaluation_point (dict, optional): Optional point for evaluating expressions
        
    Returns:
        dict: Computed tensor values and metadata
    """
    try:
        logger.info(f"Starting symbolic calculation for {dimension}D metric")
        
        # 1) instantiate the symbolic engine
        engine = ComputeTensor(coords, metric, dimension, evaluation_point)
        
        # 2) calculate all tensors in one call
        g, Gamma, R_abcd, Ricci, Scalar = engine.compute_all()
        
        # 3) format matrices as string representations
        def to_str_mat(m):
            """Convert a sympy matrix to a 2D array of strings"""
            if m is None:
                return []
            return [[str(m[i, j]) for j in range(m.shape[1])] 
                    for i in range(m.shape[0])]
        
        # Format Christoffel symbols (3D array)
        christoffel = [[[str(Gamma[a][b][c]) 
                         for c in range(dimension)] 
                        for b in range(dimension)] 
                       for a in range(dimension)]
        
        # Format Riemann tensor (4D array)
        riemann = [[[[str(R_abcd[r][s][u][v]) 
                      for v in range(dimension)]
                     for u in range(dimension)]
                    for s in range(dimension)]
                   for r in range(dimension)]
        
        # Format Ricci tensor (2D array)
        ricci_list = to_str_mat(Ricci)
        
        # Calculate Einstein tensor if not provided
        # The Einstein tensor is G_μν = R_μν - (1/2) * R * g_μν
        # where R_μν is the Ricci tensor, R is the scalar curvature, and g_μν is the metric
        einstein_tensor = []
        try:
            import sympy as sp
            if Ricci is not None and Scalar is not None and g is not None:
                einstein = Ricci - sp.Rational(1, 2) * Scalar * g
                einstein_tensor = to_str_mat(einstein)
        except Exception as e:
            logger.error(f"Error calculating Einstein tensor: {str(e)}")
        
        logger.info("Symbolic calculation completed successfully")
        
        # 4) build the response with all calculated tensors
        return {
            "success": True,
            "dimension": dimension,
            "coordinates": coords,
            "cached": False,
            "christoffelSymbols": christoffel,
            "riemannTensor": riemann,
            "ricciTensor": ricci_list,
            "scalarCurvature": str(Scalar) if Scalar is not None else "",
            "einsteinTensor": einstein_tensor,
            "weylTensor": [],  # Not implemented yet
        }
    except Exception as e:
        logger.error(f"Error in symbolic calculation: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "dimension": dimension,
            "coordinates": coords,
            "cached": False,
            "christoffelSymbols": [],
            "riemannTensor": [],
            "ricciTensor": [],
            "scalarCurvature": "",
            "einsteinTensor": [],
            "weylTensor": [],
        }
