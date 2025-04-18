from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import json
import sympy as sp
import numpy as np
import traceback
import logging
from myproject.utils.differential_operators.symbolic import (
    calculate_christoffel_symbols,
    gradient, divergence, curl, laplacian, covariant_derivative
)
from myproject.utils.differential_operators.consistency_checks import (
    check_christoffel_symmetry,
    check_metric_compatibility
)
from myproject.utils.mhd import (
    MHDSystem,
    orszag_tang_vortex,
    magnetic_rotor,
    mhd_blast_wave,
    mhd_shock_tube,
    kelvin_helmholtz_mhd,
    initialize_face_centered_b,
    compute_emf,
    update_face_centered_b,
    face_to_cell_centered_b,
    check_divergence_free
)
import base64
import io
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid GUI dependency

logger = logging.getLogger(__name__)

@api_view(['POST'])
def differential_operators(request):
    """
    Calculate differential operators using pre-calculated tensors.
    """
    logger.info("Differential operators request received")
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            calculation_mode = data.get('calculation_mode', 'symbolic')
            metric = data.get('metric', [])
            christoffel_symbols = data.get('christoffel_symbols', [])
            coordinates = data.get('coordinates', [])
            vector_field = data.get('vector_field', [])
            scalar_field = data.get('scalar_field', '')
            selected_operators = data.get('selected_operators', [])
            enable_consistency_checks = data.get('enable_consistency_checks', False)
            
            logger.info(f"Processing differential operators request: mode={calculation_mode}, operators={selected_operators}")
            
            # Initialize result dictionary
            result = {}
            
            # Choose between symbolic and numeric calculation modes
            if calculation_mode == 'symbolic':
                # Import symbolic modules
                from myproject.utils.differential_operators.symbolic import (
                    gradient, 
                    divergence, 
                    curl, 
                    laplacian, 
                    covariant_derivative,
                )
                from myproject.utils.differential_operators.consistency_checks import (
                    check_christoffel_symmetry,
                    check_metric_compatibility
                )
                
                # Convert inputs to sympy objects for symbolic calculations
                try:
                    # Convert coordinates to sympy symbols
                    coord_symbols = sp.symbols(coordinates)
                    
                    # Convert scalar field to sympy expression if provided
                    if scalar_field:
                        scalar_expr = sp.sympify(scalar_field)
                    else:
                        scalar_expr = None
                        
                    # Convert vector field components to sympy expressions if provided
                    vector_expr = []
                    if vector_field and any(vector_field):
                        for component in vector_field:
                            if component and component.strip():
                                vector_expr.append(sp.sympify(component))
                            else:
                                vector_expr.append(sp.S.Zero)  # SymPy zero
                    
                    # Convert metric to sympy matrix
                    metric_matrix = sp.Matrix(metric)
                   # … after metric_matrix = sp.Matrix(metric)
                    coord_symbols    = sp.symbols(coordinates)
                    christoffel_calc = calculate_christoffel_symbols(coord_symbols, metric_matrix)

                    # Process selected operators
                    if 'gradient' in selected_operators and scalar_expr:
                        grad_result = gradient(
                            scalar_expr,
                            coordinates,
                            metric_matrix,
                            christoffel_calc
                        )
                        result['gradient'] = [str(c) for c in grad_result]

                    if 'divergence' in selected_operators and vector_expr:
                        div_result = divergence(
                            vector_expr,
                            coordinates,
                            metric_matrix,
                            christoffel_calc
                        )
                        result['divergence'] = str(div_result)

                    if 'curl' in selected_operators and vector_expr:
                        if len(coordinates) != 3:
                            result['curl_error'] = "Curl operation requires 3D space"
                        else:
                            curl_result = curl(
                                vector_expr,
                                coordinates,
                                metric_matrix,
                                christoffel_calc
                            )
                            result['curl'] = [str(c) for c in curl_result]

                    if 'laplacian' in selected_operators and scalar_expr:
                        lap_result = laplacian(
                            scalar_expr,
                            coordinates,
                            metric_matrix,
                            christoffel_calc
                        )
                        result['laplacian'] = str(lap_result)

                    if 'covariant-derivative' in selected_operators and vector_expr:
                        cov_result = covariant_derivative(
                            vector_expr,
                            coordinates,
                            metric_matrix,
                            christoffel_calc
                        )
                        result['covariantDerivative'] = [
                            [str(comp) for comp in row]
                            for row in cov_result
                        ]

                    # consistency‑checks should line up with the above `if` blocks:
                    if enable_consistency_checks:
                        dimension = len(coordinates)
                        christoffel_dict = {}
                        for k in range(dimension):
                            for i in range(dimension):
                                for j in range(dimension):
                                    val = christoffel_calc[k][i][j]
                                    if val != 0:
                                        christoffel_dict[(k, i, j)] = val

                        result['consistencyChecks'] = {
                            'christoffelSymmetry': check_christoffel_symmetry(
                                christoffel_dict, dimension
                            ),
                            'metricCompatibility': check_metric_compatibility(
                                metric_matrix, christoffel_dict, coord_symbols, dimension
                            )
                        }

                        
                except Exception as e:
                    logger.error(f"Error in symbolic calculations: {str(e)}")
                    logger.error(traceback.format_exc())
                    result['error'] = f"Symbolic calculation error: {str(e)}"
                    
            else:  # Numeric mode
                # Import numeric modules
                from myproject.utils.differential_operators.numeric import (
                    evaluate_gradient, 
                    evaluate_divergence, 
                    evaluate_curl, 
                    evaluate_laplacian,
                    
                )
                
                try:
                    # Convert inputs to numeric arrays for calculations
                    
                    # Convert metric to numpy array
                    metric_array = np.array(metric, dtype=float)
                    
                    # Create an evaluation point (using point [1,1,1,...] for now)
                    # In a complete implementation, this would come from the request
                    eval_point = np.ones(len(coordinates))
                    
                    # Create a small grid around the evaluation point
                    # This is a simplified approach for demonstration
                    grid = [np.linspace(p * 0.9, p * 1.1, 3) for p in eval_point]
                    
                    # Convert scalar field to numpy function if provided
                    if scalar_field:
                        # Parse and evaluate the scalar field expression
                        scalar_func = lambda *args: eval(scalar_field, 
                                                      {coord: arg for coord, arg in zip(coordinates, args)})
                        # Create a grid function
                        scalar_grid = np.zeros([3] * len(coordinates))
                    else:
                        scalar_func = None
                        scalar_grid = None
                        
                    # Convert vector field components to numpy functions if provided
                    vector_func = []
                    vector_grid = []
                    if vector_field and any(vector_field):
                        for component in vector_field:
                            if component and component.strip():
                                vector_func.append(
                                    lambda *args, expr=component: eval(expr, 
                                                              {coord: arg for coord, arg in zip(coordinates, args)})
                                )
                                vector_grid.append(np.zeros([3] * len(coordinates)))
                            else:
                                vector_func.append(lambda *args: 0)
                                vector_grid.append(np.zeros([3] * len(coordinates)))
                    
                    # Calculate inverse metric
                    metric_inverse = np.linalg.inv(metric_array)
                    
                    # Process selected operators
                    if 'gradient' in selected_operators and scalar_func:
                        # Simplified implementation for now
                        grad_result = [f"∂({scalar_field})/∂{coord}" for coord in coordinates]
                        result['gradient'] = grad_result
                        
                    if 'divergence' in selected_operators and vector_func:
                        # Simplified implementation for now
                        div_terms = [f"∂({comp})/∂{coord}" for comp, coord in zip(vector_field, coordinates)]
                        result['divergence'] = " + ".join(div_terms)
                        
                    if 'curl' in selected_operators and vector_func:
                        # Curl only makes sense in 3D
                        if len(coordinates) != 3:
                            result['curl_error'] = "Curl operation requires 3D space"
                        else:
                            # Simplified implementation for now
                            result['curl'] = [
                                f"∂({vector_field[2]})/∂{coordinates[1]} - ∂({vector_field[1]})/∂{coordinates[2]}",
                                f"∂({vector_field[0]})/∂{coordinates[2]} - ∂({vector_field[2]})/∂{coordinates[0]}",
                                f"∂({vector_field[1]})/∂{coordinates[0]} - ∂({vector_field[0]})/∂{coordinates[1]}"
                            ]
                        
                    if 'laplacian' in selected_operators and scalar_func:
                        # Simplified implementation for now
                        lap_terms = [f"∂²({scalar_field})/∂{coord}²" for coord in coordinates]
                        result['laplacian'] = " + ".join(lap_terms)
                        
                    if 'covariant-derivative' in selected_operators and vector_func:
                        # Simplified implementation for now
                        cov_result = [[f"∂({vf})/∂{coord}" for vf in vector_field] for coord in coordinates]
                        result['covariantDerivative'] = cov_result
                    
                except Exception as e:
                    logger.error(f"Error in numeric calculations: {str(e)}")
                    logger.error(traceback.format_exc())
                    result['error'] = f"Numeric calculation error: {str(e)}"
            
            return Response(result)
        except Exception as e:
            logger.error(f"General error in differential operators endpoint: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({'error': str(e)}, status=400)
    else:
        return Response({'error': 'Only POST requests are allowed'}, status=405)

# Add MHD API views
@api_view(['POST'])
def mhd_simulation(request):
    """
    Run MHD simulations based on provided parameters.
    """
    logger.info("MHD simulation request received")
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Extract parameters
            simulation_type = data.get('simulation_type', 'orszag_tang')
            domain_size = data.get('domain_size', [[0, 1], [0, 1]])
            resolution = data.get('resolution', [128, 128])
            gamma = data.get('gamma', 5/3)
            final_time = data.get('final_time', 0.5)
            frame_count = data.get('frame_count', 10)
            
            # Coordinate system information
            coord_system = data.get('coordinate_system', {
                'coordinates': ['x', 'y'],
                'transformation': None  # Default to Cartesian
            })
            
            # Create MHD system based on simulation type
            if simulation_type == 'orszag_tang':
                mhd_system = orszag_tang_vortex(domain_size, resolution, gamma)
            elif simulation_type == 'magnetic_rotor':
                mhd_system = magnetic_rotor(domain_size, resolution, gamma)
            elif simulation_type == 'mhd_blast_wave':
                mhd_system = mhd_blast_wave(domain_size, resolution, gamma)
            elif simulation_type == 'shock_tube':
                mhd_system = mhd_shock_tube(domain_size, resolution, gamma)
            elif simulation_type == 'kelvin_helmholtz':
                mhd_system = kelvin_helmholtz_mhd(domain_size, resolution, gamma)
            else:
                return Response({
                    'error': f"Unknown simulation type: {simulation_type}"
                }, status=400)
            
            # Store frames for visualization
            frames = []
            
            # Output callback function to store data at regular intervals
            def save_frame(mhd, time):
                # Extract the fields to send to frontend
                frame_data = {
                    'time': float(time),
                    'density': mhd.density.tolist(),
                    'pressure': mhd.pressure.tolist(),
                    'magnetic_field': [field.tolist() for field in mhd.magnetic_field],
                    'velocity': [field.tolist() for field in mhd.velocity],
                    'max_div_b': float(mhd.check_divergence_free())
                }
                frames.append(frame_data)
            
            # Run the simulation
            output_interval = final_time / (frame_count - 1) if frame_count > 1 else final_time
            mhd_system.evolve(final_time, output_callback=save_frame, output_interval=output_interval)
            
            # Return the simulation results
            return Response({
                'success': True,
                'frames': frames,
                'simulation_type': simulation_type,
                'domain_size': domain_size,
                'resolution': resolution,
                'final_time': final_time
            })
            
        except Exception as e:
            logger.error(f"Error in MHD simulation: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({
                'error': f"MHD simulation error: {str(e)}"
            }, status=500)

@api_view(['POST'])
def mhd_snapshot(request):
    """
    Generate a visualization snapshot of the MHD state.
    """
    logger.info("MHD snapshot request received")
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Extract parameters for visualization
            field_type = data.get('field_type', 'density')
            simulation_data = data.get('simulation_data', None)
            
            if not simulation_data:
                return Response({
                    'error': "No simulation data provided"
                }, status=400)
            
            # Extract field data based on requested type
            time = simulation_data.get('time', 0.0)
            
            plt.figure(figsize=(8, 6))
            field = None
            
            if field_type == 'density':
                field = np.array(simulation_data.get('density', []))
                title = f"Density at t = {time:.3f}"
            elif field_type == 'pressure':
                field = np.array(simulation_data.get('pressure', []))
                title = f"Pressure at t = {time:.3f}"
            elif field_type.startswith('velocity'):
                component = int(field_type[-1]) if len(field_type) > 8 else 0
                velocity_components = simulation_data.get('velocity', [])
                if component < len(velocity_components):
                    field = np.array(velocity_components[component])
                    title = f"Velocity {['x', 'y', 'z'][component]} at t = {time:.3f}"
            elif field_type.startswith('magnetic'):
                component = int(field_type[-1]) if len(field_type) > 8 else 0
                magnetic_components = simulation_data.get('magnetic_field', [])
                if component < len(magnetic_components):
                    field = np.array(magnetic_components[component])
                    title = f"Magnetic Field {['x', 'y', 'z'][component]} at t = {time:.3f}"
            
            if field is None or field.size == 0:
                return Response({
                    'error': f"Invalid field type: {field_type} or empty data"
                }, status=400)
            
            # Generate the visualization
            plt.imshow(field, cmap='viridis', origin='lower', aspect='auto')
            plt.colorbar(label=field_type)
            plt.title(title)
            
            # Save the plot to a buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            # Return the base64-encoded image
            return Response({
                'success': True,
                'image_data': plot_data,
                'field_type': field_type,
                'time': time
            })
            
        except Exception as e:
            logger.error(f"Error in MHD snapshot: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({
                'error': f"MHD snapshot error: {str(e)}"
            }, status=500) 