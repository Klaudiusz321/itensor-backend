"""
Serialization utilities for ensuring data can be properly serialized to JSON.

This module provides functions to handle various data types that are not directly
JSON serializable, such as NumPy arrays, complex numbers, datetime objects, and
custom Python objects.
"""

import numpy as np
import logging
import datetime
from decimal import Decimal
import math
from myproject.utils.mhd.sanitization import sanitize_array, detect_and_fix_mhd_issues

logger = logging.getLogger(__name__)

def ensure_json_serializable(data, max_depth=20, current_depth=0, aggressive_repair=True):
    """
    Recursively processes data to ensure it can be serialized to JSON.
    
    Handles:
    - NumPy arrays and scalars (repairs NaN/Inf instead of converting to null)
    - Complex numbers (converts to {real, imag} dict)
    - Datetime objects (converts to ISO format strings)
    - Sets (converts to lists)
    - Non-string dictionary keys (converts to strings)
    - Custom objects (attempts str() conversion)
    
    Args:
        data: The data to process
        max_depth: Maximum recursion depth to prevent stack overflow
        current_depth: Current recursion depth (used internally)
        aggressive_repair: Whether to aggressively repair numeric arrays (vs. just converting to null)
        
    Returns:
        JSON serializable version of the data
    """
    if current_depth > max_depth:
        logger.warning(f"Max recursion depth ({max_depth}) reached during JSON serialization")
        return str(data)
    
    # Handle None
    if data is None:
        return None
    
    # Handle basic JSON serializable types
    if isinstance(data, (bool, int, str)):
        return data
        
    # Handle float with special cases for NaN/Inf
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            # Instead of returning null, use a reasonable default value
            if math.isnan(data):
                return 0.0
            elif math.isinf(data) and data > 0:
                return 1.0e6  # Large positive value
            else:
                return -1.0e6  # Large negative value
        return data
    
    # Handle complex numbers
    if isinstance(data, complex):
        real_part = data.real if not (math.isnan(data.real) or math.isinf(data.real)) else 0.0
        imag_part = data.imag if not (math.isnan(data.imag) or math.isinf(data.imag)) else 0.0
        return {
            "real": real_part,
            "imag": imag_part,
            "__complex__": True
        }
    
    # Handle NumPy arrays - this is where most issues occur
    if isinstance(data, np.ndarray):
        # Repair arrays instead of just filtering out bad values
        if aggressive_repair:
            # Handle complex arrays specially
            if np.issubdtype(data.dtype, np.complexfloating):
                # Repair real and imaginary parts separately
                real_part = sanitize_array(data.real, repair_mode='interpolate')
                imag_part = sanitize_array(data.imag, repair_mode='interpolate')
                # Recombine
                sanitized_data = real_part + 1j * imag_part
            else:
                # Regular array - repair directly
                sanitized_data = sanitize_array(data, repair_mode='interpolate')
                
            # If sanitization still leaves non-finite values, try more aggressive repair
            if np.any(~np.isfinite(sanitized_data)):
                sanitized_data, _, _ = detect_and_fix_mhd_issues(sanitized_data, repair_mode='aggressive')
                
            # Convert to list for JSON serialization
            return ensure_json_serializable(sanitized_data.tolist(), max_depth, current_depth + 1, aggressive_repair)
        else:
            # Original behavior - just convert to list
            return ensure_json_serializable(data.tolist(), max_depth, current_depth + 1, aggressive_repair)
    
    # Handle NumPy scalars
    if isinstance(data, np.number):
        if np.isnan(data) or np.isinf(data):
            if np.isnan(data):
                return 0.0
            elif np.isposinf(data):
                return 1.0e6
            else:
                return -1.0e6
        return float(data)
    
    # Handle NumPy booleans
    if isinstance(data, np.bool_):
        return bool(data)
    
    # Handle lists and tuples
    if isinstance(data, (list, tuple)):
        return [ensure_json_serializable(item, max_depth, current_depth + 1, aggressive_repair) for item in data]
    
    # Handle sets
    if isinstance(data, set):
        return [ensure_json_serializable(item, max_depth, current_depth + 1, aggressive_repair) for item in data]
    
    # Handle dictionaries
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # Convert non-string keys to strings
            if not isinstance(key, str):
                key = str(key)
            result[key] = ensure_json_serializable(value, max_depth, current_depth + 1, aggressive_repair)
        return result
    
    # Handle decimal.Decimal
    if isinstance(data, Decimal):
        return float(data)
    
    # Handle datetime objects
    if isinstance(data, (datetime.datetime, datetime.date)):
        try:
            return data.isoformat()
        except Exception as e:
            logger.warning(f"Failed to serialize datetime object: {e}")
            return str(data)
    
    # Handle other objects by trying to convert to string
    try:
        return str(data)
    except Exception as e:
        logger.error(f"Failed to serialize object of type {type(data)}: {e}")
        return f"<Unserializable object of type {type(data).__name__}>" 