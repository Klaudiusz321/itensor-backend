"""
Sanitization utilities for MHD simulation data.

This module provides functions to sanitize and repair MHD simulation data,
particularly handling common issues like NaN and Inf values, outliers,
and boundary artifacts.
"""

import numpy as np
import logging
from scipy import ndimage

logger = logging.getLogger(__name__)

def sanitize_array(arr, repair_mode='mean', replace_value=0.0, max_value=1e6, min_value=-1e6):
    """
    Sanitize a NumPy array by replacing NaN and Inf values with finite values.
    
    Args:
        arr: Input array or list-like object
        repair_mode: How to replace NaN/Inf values ('zero', 'mean', 'median', 'interpolate')
        replace_value: Value to use when repair_mode is 'zero'
        max_value: Maximum allowed value (clips values higher than this)
        min_value: Minimum allowed value (clips values lower than this)
        
    Returns:
        Sanitized array with all finite values
    """
    # Handle non-array inputs
    if not isinstance(arr, np.ndarray):
        try:
            # Try to convert to numpy array
            arr = np.array(arr, dtype=np.float64)
        except:
            # If conversion fails, return as is and let ensure_json_serializable handle it
            return arr
    
    # Handle empty arrays
    if arr.size == 0:
        return arr
        
    # Handle complex arrays
    if np.issubdtype(arr.dtype, np.complexfloating):
        real_part = sanitize_array(arr.real, repair_mode, replace_value, max_value, min_value)
        imag_part = sanitize_array(arr.imag, repair_mode, replace_value, max_value, min_value)
        return real_part + 1j * imag_part
        
    # Copy the array to avoid modifying the original
    sanitized = arr.copy()
    
    # Create a mask for invalid values
    invalid_mask = ~np.isfinite(sanitized)
    
    # If there are no invalid values, just clip to range and return
    if not np.any(invalid_mask):
        return np.clip(sanitized, min_value, max_value)
    
    # Special handling for when ALL values are invalid
    if np.all(invalid_mask):
        return np.zeros_like(sanitized)
    
    # Replace invalid values based on repair mode
    if repair_mode == 'zero':
        sanitized[invalid_mask] = replace_value
    elif repair_mode == 'mean':
        valid_values = sanitized[~invalid_mask]
        mean_value = np.mean(valid_values)
        sanitized[invalid_mask] = mean_value
    elif repair_mode == 'median':
        valid_values = sanitized[~invalid_mask]
        median_value = np.median(valid_values)
        sanitized[invalid_mask] = median_value
    elif repair_mode == 'interpolate' and arr.ndim <= 2:
        # Use nearest-neighbor interpolation for 1D and 2D arrays
        from scipy.interpolate import NearestNDInterpolator
        
        # Get coordinates of valid and invalid points
        if arr.ndim == 1:
            valid_indices = np.where(~invalid_mask)[0]
            invalid_indices = np.where(invalid_mask)[0]
            
            if len(valid_indices) > 0 and len(invalid_indices) > 0:
                # Create interpolator with valid points
                interpolator = NearestNDInterpolator(
                    valid_indices.reshape(-1, 1),
                    sanitized[valid_indices]
                )
                
                # Interpolate invalid points
                sanitized[invalid_indices] = interpolator(invalid_indices.reshape(-1, 1))
        elif arr.ndim == 2:
            valid_y, valid_x = np.where(~invalid_mask)
            invalid_y, invalid_x = np.where(invalid_mask)
            
            if len(valid_y) > 0 and len(invalid_y) > 0:
                # Create interpolator with valid points
                valid_points = np.column_stack((valid_y, valid_x))
                interpolator = NearestNDInterpolator(
                    valid_points,
                    sanitized[valid_y, valid_x]
                )
                
                # Interpolate invalid points
                invalid_points = np.column_stack((invalid_y, invalid_x))
                sanitized[invalid_y, invalid_x] = interpolator(invalid_points)
    else:
        # Default to median replacement
        valid_values = sanitized[~invalid_mask]
        median_value = np.median(valid_values)
        sanitized[invalid_mask] = median_value
    
    # Clip extreme values that might cause issues
    return np.clip(sanitized, min_value, max_value)

def detect_and_fix_mhd_issues(field_data, field_type='default', repair_mode='moderate', max_value=1e6, min_value=-1e6):
    """
    More advanced repair function for MHD field data with various issues.
    
    This function applies multiple repair strategies based on the repair_mode:
    - 'gentle': Only fix NaN/Inf values with minimal changes
    - 'moderate': Fix NaN/Inf and smooth extreme oscillations
    - 'aggressive': Apply more extensive repairs including boundary fixing
    
    Args:
        field_data: MHD field array to repair (density, pressure, velocity, etc.)
        field_type: Type of field ('density', 'pressure', 'velocity', 'magnetic')
        repair_mode: How aggressive to be with repairs ('gentle', 'moderate', 'aggressive')
        max_value: Maximum allowed value
        min_value: Minimum allowed value
        
    Returns:
        tuple: (repaired_data, issues_dict, repair_stats)
    """
    # Handle non-array inputs
    if not isinstance(field_data, np.ndarray):
        try:
            field_data = np.array(field_data, dtype=np.float64)
        except:
            # If conversion fails, return as is with error
            return field_data, {'error': 'Could not convert to numpy array'}, {'success': False}
    
    # Copy input to avoid modifying original
    repaired = field_data.copy()
    
    # Dictionary to track issues found
    issues = {}
    
    # Statistics about repairs
    stats = {
        'num_nan': 0,
        'num_inf': 0,
        'num_extreme': 0,
        'num_fixed': 0,
        'max_before': float(np.nanmax(repaired)) if not np.all(np.isnan(repaired)) else 0,
        'min_before': float(np.nanmin(repaired)) if not np.all(np.isnan(repaired)) else 0,
        'success': True
    }
    
    # Apply field-specific constraints
    if field_type == 'density':
        min_value = max(min_value, 1e-6)  # Density must be positive
    elif field_type == 'pressure':
        min_value = max(min_value, 1e-6)  # Pressure must be positive
    
    # Check for NaN values
    nan_mask = np.isnan(repaired)
    num_nan = np.count_nonzero(nan_mask)
    if num_nan > 0:
        issues['has_nan'] = True
        stats['num_nan'] = int(num_nan)
    
    # Check for Inf values
    inf_mask = np.isinf(repaired)
    num_inf = np.count_nonzero(inf_mask)
    if num_inf > 0:
        issues['has_inf'] = True
        stats['num_inf'] = int(num_inf)
    
    # First basic fix - replace NaN and Inf
    invalid_mask = nan_mask | inf_mask
    num_invalid = np.count_nonzero(invalid_mask)
    
    if num_invalid > 0:
        if repair_mode == 'gentle':
            # For gentle mode, just replace with zeros or reasonable values
            if field_type in ['density', 'pressure']:
                repaired[invalid_mask] = min_value  # Small positive value
            else:
                repaired[invalid_mask] = 0.0
        else:
            # Use more advanced interpolation/extrapolation
            if num_invalid < 0.8 * repaired.size:  # If we have at least 20% valid data
                # Create a mask of valid values
                valid_mask = ~invalid_mask
                
                # Get valid values and coordinates
                if repaired.ndim == 1:
                    coords = np.where(valid_mask)[0]
                    values = repaired[valid_mask]
                    
                    # Interpolate all positions
                    from scipy.interpolate import interp1d
                    
                    # Create interpolator
                    f = interp1d(coords, values, 
                                bounds_error=False, 
                                fill_value=(values[0], values[-1]))
                    
                    # Generate all indices
                    all_coords = np.arange(repaired.size)
                    repaired = f(all_coords)
                    
                elif repaired.ndim == 2:
                    # 2D case - use nearest neighbor interpolation
                    from scipy.interpolate import NearestNDInterpolator
                    
                    valid_y, valid_x = np.where(valid_mask)
                    
                    if len(valid_y) > 0:
                        # Create interpolator with valid points
                        valid_points = np.column_stack((valid_y, valid_x))
                        interpolator = NearestNDInterpolator(
                            valid_points,
                            repaired[valid_y, valid_x]
                        )
                        
                        # Get coordinates of all points
                        y_coords, x_coords = np.mgrid[0:repaired.shape[0], 0:repaired.shape[1]]
                        all_points = np.column_stack((y_coords.ravel(), x_coords.ravel()))
                        
                        # Interpolate all points
                        repaired = interpolator(all_points).reshape(repaired.shape)
            else:
                # Too many invalid values - use median or constant
                if field_type in ['density', 'pressure']:
                    repaired = np.ones_like(repaired) * min_value
                else:
                    repaired = np.zeros_like(repaired)
    
    # For moderate and aggressive repair modes, check for extreme oscillations
    if repair_mode in ['moderate', 'aggressive']:
        # Identify extreme values
        # For density/pressure, check for unusually high values
        if field_type in ['density', 'pressure']:
            median_value = np.median(repaired)
            mad = np.median(np.abs(repaired - median_value))  # Median Absolute Deviation
            extreme_threshold = median_value + 10 * mad
            extreme_mask = repaired > extreme_threshold
            
            stats['num_extreme'] += int(np.count_nonzero(extreme_mask))
            
            # Replace extreme values with local median
            if np.any(extreme_mask) and repaired.ndim == 2:
                # Use a median filter to smooth extreme values
                kernel_size = 3  # 3x3 kernel
                repaired[extreme_mask] = ndimage.median_filter(
                    repaired, size=kernel_size
                )[extreme_mask]
        
        # For velocity/magnetic fields, check for extreme gradients
        elif field_type in ['velocity', 'magnetic']:
            if repaired.ndim == 2:
                # Compute gradients
                gy, gx = np.gradient(repaired)
                grad_mag = np.sqrt(gy**2 + gx**2)
                
                # Find median and threshold
                median_grad = np.median(grad_mag)
                mad_grad = np.median(np.abs(grad_mag - median_grad))
                extreme_grad_threshold = median_grad + 10 * mad_grad
                
                # Identify cells with extreme gradients
                extreme_grad_mask = grad_mag > extreme_grad_threshold
                stats['num_extreme'] += int(np.count_nonzero(extreme_grad_mask))
                
                # Smooth areas with extreme gradients
                if np.any(extreme_grad_mask):
                    smoothed = ndimage.gaussian_filter(repaired, sigma=1.0)
                    # Only replace the cells with extreme gradients
                    repaired[extreme_grad_mask] = smoothed[extreme_grad_mask]
    
    # For aggressive repair, address boundary artifacts
    if repair_mode == 'aggressive' and repaired.ndim == 2:
        # Check for artifacts at boundaries
        boundary_width = 2
        
        # Get interior region
        interior = repaired[boundary_width:-boundary_width, boundary_width:-boundary_width]
        
        # Get statistics of interior
        interior_median = np.median(interior)
        interior_mad = np.median(np.abs(interior - interior_median))
        
        # Check all boundaries
        for i in range(boundary_width):
            # Top boundary
            top = repaired[i, :]
            top_mask = np.abs(top - interior_median) > 5 * interior_mad
            if np.any(top_mask):
                repaired[i, top_mask] = interior_median
            
            # Bottom boundary
            bottom = repaired[-(i+1), :]
            bottom_mask = np.abs(bottom - interior_median) > 5 * interior_mad
            if np.any(bottom_mask):
                repaired[-(i+1), bottom_mask] = interior_median
            
            # Left boundary
            left = repaired[:, i]
            left_mask = np.abs(left - interior_median) > 5 * interior_mad
            if np.any(left_mask):
                repaired[:, i][left_mask] = interior_median
            
            # Right boundary
            right = repaired[:, -(i+1)]
            right_mask = np.abs(right - interior_median) > 5 * interior_mad
            if np.any(right_mask):
                repaired[:, -(i+1)][right_mask] = interior_median
    
    # Final clip to ensure values are within bounds
    repaired = np.clip(repaired, min_value, max_value)
    
    # Check if we've fixed all issues
    if np.any(~np.isfinite(repaired)):
        # If we still have non-finite values, use brute force approach
        repaired = np.nan_to_num(repaired, nan=0.0, posinf=max_value, neginf=min_value)
    
    # Calculate repair statistics
    stats['num_fixed'] = stats['num_nan'] + stats['num_inf'] + stats['num_extreme']
    stats['max_after'] = float(np.max(repaired))
    stats['min_after'] = float(np.min(repaired))
    
    return repaired, issues, stats

def repair_mhd_inconsistencies(density, pressure, velocity, magnetic_field, gamma=5/3):
    """
    Check and repair inconsistencies between MHD primitive variables to ensure they meet physical constraints.
    
    This function enforces:
    1. Positive density
    2. Positive pressure
    3. Reasonable velocity magnitudes
    4. Energy positivity condition
    
    Args:
        density: Density field (array-like)
        pressure: Pressure field (array-like)
        velocity: List of velocity component arrays
        magnetic_field: List of magnetic field component arrays
        gamma: Adiabatic index
        
    Returns:
        tuple: (density, pressure, velocity, magnetic_field) - repaired fields
    """
    # Convert inputs to numpy arrays if they aren't already
    if not isinstance(density, np.ndarray):
        density = np.array(density, dtype=np.float64)
    
    if not isinstance(pressure, np.ndarray):
        pressure = np.array(pressure, dtype=np.float64)
    
    velocity_arrays = []
    for v in velocity:
        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype=np.float64)
        velocity_arrays.append(v)
    
    magnetic_arrays = []
    for b in magnetic_field:
        if not isinstance(b, np.ndarray):
            b = np.array(b, dtype=np.float64)
        magnetic_arrays.append(b)
    
    # Step 1: Ensure positive density
    density = np.maximum(density, 1e-6)
    
    # Step 2: Calculate kinetic energy
    v_squared = np.zeros_like(density)
    for v in velocity_arrays:
        v_squared += v**2
    
    # Step 3: Calculate magnetic energy
    b_squared = np.zeros_like(density)
    for b in magnetic_arrays:
        b_squared += b**2
    
    # Step 4: Enforce velocity bounds - limit to a reasonable Mach number (e.g., 100)
    # First, calculate sound speed
    sound_speed = np.sqrt(gamma * pressure / density)
    
    # Get velocity magnitude
    v_mag = np.sqrt(v_squared)
    
    # Where velocity magnitude is too high, rescale it
    max_mach = 100  # Maximum allowed Mach number
    max_v = max_mach * sound_speed
    
    excessive_v = v_mag > max_v
    if np.any(excessive_v):
        # Create scaling factor where velocity is excessive
        scale = np.ones_like(density)
        # Only apply scaling where needed
        scale[excessive_v] = max_v[excessive_v] / v_mag[excessive_v]
        
        # Apply scaling to all velocity components
        for i in range(len(velocity_arrays)):
            velocity_arrays[i] = velocity_arrays[i] * scale
        
        # Recalculate v_squared
        v_squared = np.zeros_like(density)
        for v in velocity_arrays:
            v_squared += v**2
    
    # Step 5: Ensure pressure positivity with energy constraint
    # MHD total energy = internal energy + kinetic energy + magnetic energy
    # E = p/(gamma-1) + 0.5*rho*v^2 + B^2/2
    
    # Minimum pressure needed for positivity
    min_pressure = 1e-6
    
    # Ensure pressure is positive
    pressure = np.maximum(pressure, min_pressure)
    
    return density, pressure, velocity_arrays, magnetic_arrays 