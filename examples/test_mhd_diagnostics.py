#!/usr/bin/env python
"""
Test script for MHD diagnostic endpoints.

This script demonstrates how to use the MHD diagnostic endpoints
directly from Python without going through the web API.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# Add the parent directory to the Python path to import the project modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    # Import MHD modules directly
    from myproject.utils.mhd.core import magnetic_rotor_2d, orszag_tang_vortex_2d
    import numpy as np
except ImportError as e:
    print(f"Error importing MHD modules: {str(e)}")
    print("Make sure you are running this script from the project root or have added it to PYTHONPATH")
    sys.exit(1)

def run_basic_diagnostic(simulation_type="magnetic_rotor", resolution=None, gamma=5/3):
    """
    Run basic diagnostics on an MHD simulation.
    
    Args:
        simulation_type: Type of simulation ('magnetic_rotor' or 'orszag_tang_vortex')
        resolution: Grid resolution as [nx, ny]
        gamma: Adiabatic index
        
    Returns:
        Dictionary with diagnostic results
    """
    print(f"Running basic diagnostic for {simulation_type} simulation...")
    
    # Default resolution
    if resolution is None:
        resolution = [32, 32]
    
    # Standard domain size
    domain_size = [[0,1],[0,1]]
    
    # Dictionary to store diagnostic information
    diagnostics = {
        'simulation_type': simulation_type,
        'resolution': resolution,
        'gamma': gamma,
        'import_success': True,
        'initialization': {'success': False, 'error': None},
        'single_step': {'success': False, 'error': None},
        'buffer_check': {'success': False, 'error': None},
    }
    
    start_time = time.time()
    
    # Step 1: Initialize MHD system
    try:
        if simulation_type == 'orszag_tang_vortex':
            mhd_system = orszag_tang_vortex_2d(domain_size, resolution, gamma)
        else:  # magnetic_rotor
            mhd_system = magnetic_rotor_2d(domain_size, resolution, gamma)
        
        diagnostics['initialization']['success'] = True
        diagnostics['initialization']['system_info'] = {
            'resolution': mhd_system.resolution,
            'coordinate_system': str(mhd_system.coordinate_system),
            'has_buffers': hasattr(mhd_system, '_buffer'),
            'buffer_keys': list(mhd_system._buffer.keys()) if hasattr(mhd_system, '_buffer') else None,
            'dt': float(mhd_system.compute_time_step()),
            'cfl': float(mhd_system.cfl_number)
        }
        
        # Check for NaN/Inf in initial state
        initial_check = {
            'density': {
                'has_nan': bool(np.isnan(mhd_system.density).any()),
                'has_inf': bool(np.isinf(mhd_system.density).any()),
                'min': float(np.min(mhd_system.density)),
                'max': float(np.max(mhd_system.density))
            },
            'pressure': {
                'has_nan': bool(np.isnan(mhd_system.pressure).any()),
                'has_inf': bool(np.isinf(mhd_system.pressure).any()),
                'min': float(np.min(mhd_system.pressure)),
                'max': float(np.max(mhd_system.pressure))
            }
        }
        diagnostics['initialization']['initial_check'] = initial_check
        
    except Exception as e:
        import traceback
        diagnostics['initialization']['success'] = False
        diagnostics['initialization']['error'] = str(e)
        diagnostics['initialization']['traceback'] = traceback.format_exc()
        print(f"Initialization failed: {str(e)}")
        return diagnostics
    
    # Step 2: Check buffer allocation
    try:
        # Test buffer allocation if the system has buffers
        if hasattr(mhd_system, '_buffer'):
            buffer_sizes = {}
            for key, buffer in mhd_system._buffer.items():
                if hasattr(buffer, 'shape'):
                    buffer_sizes[key] = list(buffer.shape)
                elif isinstance(buffer, list) and all(hasattr(b, 'shape') for b in buffer):
                    buffer_sizes[key] = [list(b.shape) for b in buffer]
            diagnostics['buffer_check']['success'] = True
            diagnostics['buffer_check']['buffer_sizes'] = buffer_sizes
        else:
            # Create some essential buffers for testing
            density_buffer = np.zeros_like(mhd_system.density)
            fx_buffer = np.zeros_like(mhd_system.density)
            diagnostics['buffer_check']['success'] = True
            diagnostics['buffer_check']['manual_buffer_test'] = "Allocated test buffers successfully"
    except Exception as e:
        diagnostics['buffer_check']['success'] = False
        diagnostics['buffer_check']['error'] = str(e)
    
    # Step 3: Try a single time step
    try:
        # Compute safe time step
        dt = mhd_system.compute_time_step() * 0.4  # Use smaller time step for safety
        
        # Try to use safe method if available
        if hasattr(mhd_system, '_safe_advance_time_step'):
            success = mhd_system._safe_advance_time_step(dt)
            if success:
                diagnostics['single_step']['success'] = True
                diagnostics['single_step']['method'] = 'safe_advance'
                diagnostics['single_step']['dt'] = float(dt)
            else:
                # Try normal method with smaller step
                dt = dt * 0.5
                mhd_system.advance_time_step(dt)
                diagnostics['single_step']['success'] = True
                diagnostics['single_step']['method'] = 'normal_advance_after_safe_failed'
                diagnostics['single_step']['dt'] = float(dt)
        else:
            # Use normal method
            mhd_system.advance_time_step(dt)
            diagnostics['single_step']['success'] = True
            diagnostics['single_step']['method'] = 'normal_advance'
            diagnostics['single_step']['dt'] = float(dt)
        
        # Check for NaN/Inf after single step
        post_step_check = {
            'density': {
                'has_nan': bool(np.isnan(mhd_system.density).any()),
                'has_inf': bool(np.isinf(mhd_system.density).any()),
                'min': float(np.min(mhd_system.density)),
                'max': float(np.max(mhd_system.density))
            },
            'pressure': {
                'has_nan': bool(np.isnan(mhd_system.pressure).any()),
                'has_inf': bool(np.isinf(mhd_system.pressure).any()),
                'min': float(np.min(mhd_system.pressure)),
                'max': float(np.max(mhd_system.pressure))
            }
        }
        diagnostics['single_step']['post_step_check'] = post_step_check
        
    except Exception as e:
        import traceback
        diagnostics['single_step']['success'] = False
        diagnostics['single_step']['error'] = str(e)
        diagnostics['single_step']['traceback'] = traceback.format_exc()
    
    # Step 4: Memory diagnostics
    try:
        if hasattr(mhd_system, 'diagnose_memory_issues'):
            memory_diagnostics = mhd_system.diagnose_memory_issues()
            diagnostics['memory_diagnostics'] = memory_diagnostics
    except Exception as e:
        diagnostics['memory_check'] = {'success': False, 'error': str(e)}
    
    # Total execution time
    diagnostics['execution_time'] = time.time() - start_time
    
    return diagnostics

def run_stress_test(simulation_type="orszag_tang_vortex", base_resolution=None, 
                   gamma_values=None, cfl_values=None, test_duration=0.05):
    """
    Run a stress test with different parameters to identify stability issues.
    
    Args:
        simulation_type: Type of simulation ('magnetic_rotor' or 'orszag_tang_vortex')
        base_resolution: Base grid resolution as [nx, ny]
        gamma_values: List of gamma values to test
        cfl_values: List of CFL values to test
        test_duration: Duration of each test
        
    Returns:
        Dictionary with stress test results
    """
    print(f"Running stress test for {simulation_type} simulation...")
    
    # Default values
    if base_resolution is None:
        base_resolution = [32, 32]
    if gamma_values is None:
        gamma_values = [1.4, 5/3, 2.0]
    if cfl_values is None:
        cfl_values = [0.2, 0.4, 0.6]
    
    # Standard domain size
    domain_size = [[0,1],[0,1]]
    
    # Dictionary to store test results
    results = {
        'simulation_type': simulation_type,
        'base_resolution': base_resolution,
        'gamma_values': gamma_values,
        'cfl_values': cfl_values,
        'test_duration': test_duration,
        'tests': [],
        'success_rate': 0.0,
        'error_summary': {}
    }
    
    # Track error patterns
    error_counts = {}
    total_tests = 0
    successful_tests = 0
    
    # Run a matrix of tests with different parameters
    for gamma in gamma_values:
        for cfl in cfl_values:
            # Create a test configuration
            test_config = {
                'gamma': gamma,
                'cfl': cfl,
                'resolution': base_resolution,
                'result': None,
                'error': None,
                'performance': None,
                'success': False
            }
            
            try:
                # Initialize the MHD system
                start_time = time.time()
                
                if simulation_type == 'orszag_tang_vortex':
                    mhd_system = orszag_tang_vortex_2d(domain_size, base_resolution, gamma)
                else:  # magnetic_rotor
                    mhd_system = magnetic_rotor_2d(domain_size, base_resolution, gamma)
                
                # Set the CFL number
                mhd_system.cfl_number = cfl
                
                # Run a short simulation with timing
                init_time = time.time() - start_time
                
                # Compute baseline timestep
                dt = mhd_system.compute_time_step()
                steps_needed = int(test_duration / dt) + 1
                
                # Limit to at most 10 steps for quick testing
                steps = min(steps_needed, 10)
                
                # Track evolution success
                evolution_success = True
                sim_start = time.time()
                
                # Try to evolve the system
                for step in range(steps):
                    try:
                        # Use safe method if available
                        if hasattr(mhd_system, '_safe_advance_time_step'):
                            success = mhd_system._safe_advance_time_step()
                            if not success:
                                evolution_success = False
                                break
                        else:
                            mhd_system.advance_time_step()
                        
                        # Check for NaN/Inf after each step
                        if (np.isnan(mhd_system.density).any() or 
                            np.isinf(mhd_system.density).any() or
                            np.isnan(mhd_system.pressure).any() or
                            np.isinf(mhd_system.pressure).any()):
                            evolution_success = False
                            break
                            
                    except Exception as step_error:
                        evolution_success = False
                        test_config['error'] = f"Step {step} failed: {str(step_error)}"
                        # Track error pattern
                        error_key = str(step_error)[:100]  # Truncate long error messages
                        error_counts[error_key] = error_counts.get(error_key, 0) + 1
                        break
                
                sim_time = time.time() - sim_start
                
                # Record results
                if evolution_success:
                    successful_tests += 1
                    test_config['success'] = True
                    
                    # Check final state for consistency
                    max_div_b = 0.0
                    if hasattr(mhd_system, 'check_divergence_free'):
                        max_div_b = mhd_system.check_divergence_free()
                    
                    test_config['result'] = {
                        'final_time': float(mhd_system.time),
                        'steps_completed': steps,
                        'max_density': float(np.max(mhd_system.density)),
                        'min_density': float(np.min(mhd_system.density)),
                        'max_div_b': float(max_div_b)
                    }
                
                # Record performance metrics
                test_config['performance'] = {
                    'initialization_time': init_time,
                    'simulation_time': sim_time,
                    'steps_per_second': steps / sim_time if sim_time > 0 else 0
                }
                
            except Exception as e:
                test_config['error'] = str(e)
                # Track error pattern
                error_key = str(e)[:100]  # Truncate long error messages
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
            
            # Add test to results
            results['tests'].append(test_config)
            total_tests += 1
            
            # Print progress
            print(f"  Test gamma={gamma}, cfl={cfl}: {'SUCCESS' if test_config['success'] else 'FAILED'}")
    
    # Compute summary statistics
    if total_tests > 0:
        results['success_rate'] = successful_tests / total_tests
    
    # Add error summary
    results['error_summary'] = {
        'total_errors': total_tests - successful_tests,
        'most_common_errors': sorted(
            [(error, count) for error, count in error_counts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3 most common errors
    }
    
    # Add recommendation for optimal parameters
    if successful_tests > 0:
        # Find the most successful configuration
        best_config = None
        best_score = -1
        
        for test in results['tests']:
            if test['success']:
                # Simple scoring: prioritize higher CFL (faster simulation) if successful
                score = test['cfl']
                if score > best_score:
                    best_score = score
                    best_config = {
                        'gamma': test['gamma'],
                        'cfl': test['cfl'],
                        'resolution': test['resolution']
                    }
        
        if best_config:
            results['recommended_config'] = best_config
    
    return results

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description='Test MHD diagnostic endpoints.')
    parser.add_argument('--type', choices=['diagnostic', 'stress'], default='diagnostic',
                        help='Type of test to run (diagnostic or stress)')
    parser.add_argument('--simulation', choices=['orszag_tang_vortex', 'magnetic_rotor'], 
                      default='magnetic_rotor',
                      help='Simulation type')
    parser.add_argument('--resolution', type=int, nargs=2, default=[32, 32],
                      help='Resolution as "nx ny" (e.g., "32 32")')
    parser.add_argument('--gamma', type=float, default=5/3,
                      help='Adiabatic index (gamma)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    # Run the selected test
    if args.type == 'diagnostic':
        results = run_basic_diagnostic(
            simulation_type=args.simulation,
            resolution=args.resolution,
            gamma=args.gamma
        )
        print("\nDiagnostic Results Summary:")
        print(f"  Initialization: {'SUCCESS' if results['initialization']['success'] else 'FAILED'}")
        print(f"  Buffer Check:    {'SUCCESS' if results['buffer_check']['success'] else 'FAILED'}")
        print(f"  Single Step:     {'SUCCESS' if results['single_step']['success'] else 'FAILED'}")
        
    else:  # Stress test
        results = run_stress_test(
            simulation_type=args.simulation,
            base_resolution=args.resolution,
            gamma_values=[args.gamma, args.gamma * 0.9, args.gamma * 1.1],  # Test variations
            cfl_values=[0.2, 0.4, 0.6]
        )
        print("\nStress Test Results Summary:")
        print(f"  Tests run:    {len(results['tests'])}")
        print(f"  Success rate: {results['success_rate'] * 100:.1f}%")
        
        if 'recommended_config' in results:
            best = results['recommended_config']
            print("\nRecommended Configuration:")
            print(f"  gamma={best['gamma']}, cfl={best['cfl']}, resolution={best['resolution']}")
    
    # Save results to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            # Handle NumPy values by converting to Python native types
            json_results = json.dumps(results, default=lambda x: float(x) if isinstance(x, np.number) else x, indent=2)
            f.write(json_results)
        print(f"\nResults saved to {args.output}")
    
    return results

if __name__ == "__main__":
    main() 