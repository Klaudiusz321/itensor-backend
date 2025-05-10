
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import json
import sympy as sp
import numpy as np
import traceback
import logging
from rest_framework.decorators import api_view
from django.http import StreamingHttpResponse

from myproject.utils.mhd.core import (
    
    orszag_tang_vortex_2d,
    magnetic_rotor_2d
)
import base64
import io
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid GUI dependency
import hashlib
import pickle
import os
from pathlib import Path
from myproject.utils.mhd.sanitization import sanitize_array
import logging

logger = logging.getLogger(__name__)



def ensure_json_serializable(obj):
    """
    Zamienia numpy array, datetimes, Decimal itp. na listy/str/int,
    żeby DRF mógł to zakodować do JSON.
    """
    # słowniki → rekurencja
    if isinstance(obj, dict):
        return {ensure_json_serializable(k): ensure_json_serializable(v)
                for k, v in obj.items()}
    # listy/tuple → rekurencja
    if isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(v) for v in obj]
    # numpy array / pandas Series itp.
    if hasattr(obj, "tolist"):
        return obj.tolist()
    # spróbuj JSON; jeśli nie pójdzie, zamień na str()
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)

@api_view(['POST'])
def mhd_simulation(request):
    if request.method != 'POST':
        return StreamingHttpResponse(status=405)

    try:
        # Extract parameters from request
        data = json.loads(request.body)
        simulation_type = data.get('simulation_type', 'orszag_tang_vortex')
        domain_size = data.get('domain_size', [[0,1],[0,1]])
        evolution_time = data.get('evolution_time', 0.2)  # Time to evolve before returning frame
        
        # Fixed resolution of 64x64 for better performance
        resolution = [64, 64]
        logger.info(f"Running {simulation_type} simulation with fixed resolution {resolution} to time {evolution_time}")
            
        gamma = data.get('gamma', 5/3)
        
        # Check if simulation results are already cached
        cache_dir = Path("./simulation_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a unique cache key based on simulation parameters
        cache_key = f"{simulation_type}_{resolution[0]}x{resolution[1]}_{evolution_time}_{gamma}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = cache_dir / f"{cache_hash}.pkl"
        
        # Check if cached results exist
        try:
            if cache_file.exists():
                logger.info(f"Using cached results for {simulation_type} simulation")
                with open(cache_file, 'rb') as f:
                    cached_results = pickle.load(f)
                
                # Make sure cached data is JSON serializable
                cached_results = ensure_json_serializable(cached_results)
                
                # Return the cached single frame
                return Response(cached_results)
        except Exception as cache_error:
            logger.warning(f"Failed to load cache: {str(cache_error)}. Will regenerate.")

        # Initialize the MHD system based on the selected simulation type
        logger.info(f"Initializing MHD simulation of type: {simulation_type}")
        
        try:
            # Only support Orszag-Tang and Magnetic Rotor
            if simulation_type not in ['orszag_tang_vortex', 'magnetic_rotor']:
                logger.warning(f"Unsupported simulation type: {simulation_type}, defaulting to orszag_tang_vortex")
                simulation_type = 'orszag_tang_vortex'
            
            if simulation_type == 'orszag_tang_vortex':
                mhd_system = orszag_tang_vortex_2d(domain_size, resolution, gamma)
            else:  # magnetic_rotor
                mhd_system = magnetic_rotor_2d(domain_size, resolution, gamma)
            
            # Ensure max_wavespeed is initialized
            if not hasattr(mhd_system, 'max_wavespeed') or mhd_system.max_wavespeed is None:
                mhd_system.max_wavespeed = 1.0  # Default value
                
            # Optimize simulation parameters for faster performance
            if hasattr(mhd_system, 'optimize_for_demo'):
                logger.info("Applying demo mode optimizations")
                mhd_system.optimize_for_demo()
            else:
                # Fallback if optimize_for_demo is not available
                if simulation_type == 'magnetic_rotor':
                    mhd_system.cfl_number = 0.55  # Slightly increased for faster simulation
                else:
                    mhd_system.cfl_number = 0.65  # Slightly increased for faster simulation
            
            # Verify that the MHD system was properly initialized
            if not hasattr(mhd_system, 'evolve') or not callable(getattr(mhd_system, 'evolve')):
                raise ValueError(f"MHD system for '{simulation_type}' does not have a valid evolve method")
            
        except (AttributeError, TypeError) as setup_error:
            # Specific error for initialization issues
            logger.error(f"Failed to initialize {simulation_type} simulation: {str(setup_error)}")
            return Response({
                'error': f"Failed to set up the {simulation_type} simulation: {str(setup_error)}"
            }, status=500)
        
        try:
            # Make sure compute_wavespeeds method is available
            if not hasattr(mhd_system, 'compute_wavespeeds'):
                logger.warning("MHD system missing compute_wavespeeds method, adding fallback")
                # Add a fallback method that just returns a default value
                def fallback_compute_wavespeeds(self):
                    self.max_wavespeed = 5.0  # Default safe value
                    return self.max_wavespeed
                
                # Attach the method to the instance
                import types
                mhd_system.compute_wavespeeds = types.MethodType(fallback_compute_wavespeeds, mhd_system)
            
            # Evolve the system to the desired time
            logger.info(f"Evolving {simulation_type} simulation to time {evolution_time}")
            mhd_system.evolve(evolution_time)
            
            # Check for NaN or Inf values and try to repair if needed
            has_issues = False
            repair_attempt = False
            
            # Check density, pressure, velocity, and magnetic field for NaN or Inf
            if (np.any(~np.isfinite(mhd_system.density)) or 
                np.any(~np.isfinite(mhd_system.pressure)) or
                any(np.any(~np.isfinite(v)) for v in mhd_system.velocity) or
                any(np.any(~np.isfinite(b)) for b in mhd_system.magnetic_field)):
                
                has_issues = True
                logger.warning(f"NaN or Inf values detected in simulation results. Attempting repair.")
                
                # Try to repair the fields
                try:
                    # First check for obvious issues in density and pressure
                    mhd_system.density = sanitize_array(mhd_system.density, repair_mode='median')
                    mhd_system.pressure = sanitize_array(mhd_system.pressure, repair_mode='median')
                    
                    # Then handle velocity and magnetic field
                    for i in range(len(mhd_system.velocity)):
                        mhd_system.velocity[i] = sanitize_array(mhd_system.velocity[i], repair_mode='median')
                    
                    for i in range(len(mhd_system.magnetic_field)):
                        mhd_system.magnetic_field[i] = sanitize_array(mhd_system.magnetic_field[i], repair_mode='median')
                    
                    repair_attempt = True
                except Exception as repair_error:
                    logger.error(f"Failed to repair simulation data: {str(repair_error)}")
            
            # Create the result frame
            frame = {
                'time': float(evolution_time),
                'density': mhd_system.density.tolist(),
                'pressure': mhd_system.pressure.tolist(),
                'magnetic_field': [b.tolist() for b in mhd_system.magnetic_field],
                'velocity': [v.tolist() for v in mhd_system.velocity]
            }
            
            # Calculate divergence of B
            try:
                from myproject.utils.mhd.grid import numerical_divergence
                
                # Convert magnetic field to list
                magnetic_field_list = []
                for component in mhd_system.magnetic_field:
                    magnetic_field_list.append(component)
                    
                # Get metric determinant (or use identity if not available)
                metric_determinant = getattr(mhd_system, 'numeric_metric_determinant', None)
                if metric_determinant is None:
                    metric_determinant = np.ones_like(mhd_system.density)
                    
                # Compute divergence
                div_b = numerical_divergence(
                    magnetic_field_list,
                    mhd_system.grid,
                    mhd_system.coordinate_system.get('coordinates', [f'x{i}' for i in range(len(mhd_system.grid))]),
                    [mhd_system.spacing[coord] for coord in mhd_system.coordinate_system.get('coordinates', [f'x{i}' for i in range(len(mhd_system.grid))])],
                    metric_determinant
                )
                
                max_div_b = float(np.max(np.abs(div_b)))
                frame['max_div_b'] = max_div_b
                
                # Check if div(B) is well-behaved
                if np.any(~np.isfinite(div_b)) or max_div_b > 1.0:
                    logger.warning(f"High divergence of B detected: {max_div_b}")
                
            except Exception as div_error:
                logger.error(f"Failed to compute divergence of B: {str(div_error)}")
                frame['div_b_error'] = str(div_error)
                
            # Add diagnostic information
            frame['has_issues'] = has_issues
            frame['repair_attempt'] = repair_attempt
            
            # Cache the results if simulation is successful
            if not has_issues or repair_attempt:
                try:
                    # Ensure the result is serializable before caching
                    with open(cache_file, 'wb') as f:
                        pickle.dump({
                            'success': True,
                            'frame': ensure_json_serializable(frame),
                            'simulation_type': simulation_type
                        }, f)
                    logger.info(f"Cached simulation results to {cache_file}")
                except Exception as cache_error:
                    logger.error(f"Failed to cache simulation results: {str(cache_error)}")
            
            # Prepare the final response
            result = {
                'success': True,
                'frame': frame,
                'simulation_type': simulation_type
            }
            
            # Ensure the response is JSON serializable
            result = ensure_json_serializable(result)
            
            return Response(result)
            
        except Exception as sim_error:
            logger.error(f"Error during simulation: {str(sim_error)}")
            logger.error(traceback.format_exc())
            
            # Try to return partial results if available
            partial_result = {
                'success': False,
                'error': str(sim_error),
                'error_type': type(sim_error).__name__,
                'simulation_type': simulation_type
            }
            
            # If the simulation has partial data, include it
            if 'mhd_system' in locals() and mhd_system is not None:
                try:
                    frame = {
                        'time': float(getattr(mhd_system, 'time', 0.0)),
                        'partial_data': True
                    }
                    
                    # Try to collect partial data
                    if hasattr(mhd_system, 'density'):
                        frame['density'] = sanitize_array(mhd_system.density).tolist()
                    
                    if hasattr(mhd_system, 'pressure'):
                        frame['pressure'] = sanitize_array(mhd_system.pressure).tolist()
                    
                    if hasattr(mhd_system, 'velocity'):
                        frame['velocity'] = [sanitize_array(v).tolist() for v in mhd_system.velocity]
                        
                    if hasattr(mhd_system, 'magnetic_field'):
                        frame['magnetic_field'] = [sanitize_array(b).tolist() for b in mhd_system.magnetic_field]
                    
                    partial_result['frame'] = frame
                except Exception as partial_error:
                    logger.error(f"Failed to collect partial results: {str(partial_error)}")
                    partial_result['partial_data_error'] = str(partial_error)
            
            # Ensure the error response is JSON serializable
            partial_result = ensure_json_serializable(partial_result)
            
            return Response(partial_result, status=500)
            
    except Exception as e:
        logger.error(f"Unhandled error in mhd_simulation: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(ensure_json_serializable({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), status=500)

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
            
            try:
                if field_type == 'density':
                    field_data = simulation_data.get('density', [])
                elif field_type == 'pressure':
                    field_data = simulation_data.get('pressure', [])
                elif field_type.startswith('velocity'):
                    component = int(field_type[-1]) if len(field_type) > 8 else 0
                    velocity_components = simulation_data.get('velocity', [])
                    if component < len(velocity_components):
                        field_data = velocity_components[component]
                    else:
                        return Response({
                            'error': f"Velocity component index {component} is out of range, max index is {len(velocity_components) - 1}"
                        }, status=400)
                elif field_type.startswith('magnetic'):
                    component = int(field_type[-1]) if len(field_type) > 8 else 0
                    magnetic_components = simulation_data.get('magnetic_field', [])
                    if component < len(magnetic_components):
                        field_data = magnetic_components[component]
                    else:
                        return Response({
                            'error': f"Magnetic field component index {component} is out of range, max index is {len(magnetic_components) - 1}"
                        }, status=400)
                else:
                    return Response({
                        'error': f"Invalid field type: {field_type}"
                    }, status=400)
            except IndexError as e:
                logger.error(f"Index error accessing field components: {str(e)}")
                logger.error(traceback.format_exc())
                return Response({
                    'error': f"Error accessing field components: {str(e)}"
                }, status=400)
                
            try:
                # Convert to numpy array
                field = np.array(field_data, dtype=np.float64)
                
                # Sanitize field data (handle NaN, Inf, etc.)
                field = sanitize_array(field)
                
                # Create the visualization
                plt.imshow(field, cmap='viridis', origin='lower', aspect='auto')
                plt.colorbar(label=field_type)
                plt.title(f"{field_type.capitalize()} at t = {time:.3f}")
                
                # Save to buffer
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                
                # Return the image
                result = {
                    'success': True,
                    'field_type': field_type,
                    'time': time,
                    'image_data': image_data,
                    'min_value': float(np.min(field)),
                    'max_value': float(np.max(field)),
                    'mean_value': float(np.mean(field))
                }
                
                # Ensure the response is JSON serializable
                result = ensure_json_serializable(result)
                
                return Response(result)
                
            except Exception as viz_error:
                logger.error(f"Error in visualization: {str(viz_error)}")
                logger.error(traceback.format_exc())
                return Response(ensure_json_serializable({
                    'error': f"Visualization error: {str(viz_error)}"
                }), status=500)
                
        except Exception as e:
            logger.error(f"Error in MHD snapshot: {str(e)}")
            logger.error(traceback.format_exc())
            return Response(ensure_json_serializable({
                'error': f"Error generating snapshot: {str(e)}"
            }), status=500)
    
    return Response({'error': 'Only POST requests are allowed'}, status=405)

@api_view(['POST'])
def mhd_field_plots(request):
    """
    Generate visualization plots for all MHD fields after time evolution.
    Returns multiple plots as base64 encoded images.
    """
    try:
        data = request.data
        sim_type      = data.get('simulation_type', 'orszag_tang_vortex')
        domain_size   = data.get('domain_size', [[0,1],[0,1]])
        evolution_time= data.get('evolution_time', 0.2)
        gamma         = data.get('gamma', 5/3)

        # --- 1) Stwórz system MHD ---
        if sim_type == 'magnetic_rotor':
            mhd = magnetic_rotor_2d(domain_size, [64,64], gamma)
        else:
            mhd = orszag_tang_vortex_2d(domain_size, [64,64], gamma)

        # --- 2) Ewoluuj system do czasu evolution_time ---
        mhd.evolve(evolution_time)

        # --- 3) Zbierz dane wyjściowe ---
        frame = {
            'time': float(evolution_time),
            'density':         sanitize_array(mhd.density).tolist(),
            'pressure':        sanitize_array(mhd.pressure).tolist(),
            'velocity':        [sanitize_array(v).tolist() for v in mhd.velocity],
            'magnetic_field':  [sanitize_array(b).tolist() for b in mhd.magnetic_field],
        }

        # --- 4) Funkcja pomocnicza do rysowania i kodowania obrazu ---
        def make_base64_plot(field_array, title):
            plt.figure(figsize=(5,4))
            plt.imshow(field_array, origin='lower', aspect='auto')
            plt.colorbar(label=title)
            plt.title(title)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        # --- 5) Przygotuj słownik plotów ---
        plots = {}
        # Density i pressure
        plots['density'] = {
            'image_data': make_base64_plot(np.array(frame['density']), 'Density'),
            'min': float(np.min(frame['density'])),
            'max': float(np.max(frame['density']))
        }
        plots['pressure'] = {
            'image_data': make_base64_plot(np.array(frame['pressure']), 'Pressure'),
            'min': float(np.min(frame['pressure'])),
            'max': float(np.max(frame['pressure']))
        }
        # Velocity components
        for idx, vel in enumerate(frame['velocity']):
            arr = np.array(vel)
            plots[f'velocity{idx}'] = {
                'image_data': make_base64_plot(arr, f'Velocity {idx}'),
                'min': float(np.min(arr)),
                'max': float(np.max(arr))
            }
        # Magnetic field components
        for idx, mag in enumerate(frame['magnetic_field']):
            arr = np.array(mag)
            plots[f'magnetic{idx}'] = {
                'image_data': make_base64_plot(arr, f'Magnetic {idx}'),
                'min': float(np.min(arr)),
                'max': float(np.max(arr))
            }

        # --- 6) Odpowiedź ---
        return Response({
            'success': True,
            'simulation_type': sim_type,
            'time': frame['time'],
            'plots': plots
        })

    except Exception as e:
        logger.error("Error in MHD field plots:\n" + traceback.format_exc())
        return Response({'error': str(e)}, status=500)