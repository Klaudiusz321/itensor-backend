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


class Serialization:
    def __init__(self, data, max_depth=20, current_depth=0, aggressive_repair=True):
        self.data = data
        self.max_depth = max_depth
        self.current_depth = current_depth
        self.aggressive_repair = aggressive_repair

    def ensure_json_serializable(self):
        
        if self.current_depth > self.max_depth:
            logger.warning(f"Max recursion depth ({self.max_depth}) reached during JSON serialization")
            return str(self.data)
        
        # Handle None
        if self.data is None:
            return None
        
        # Handle basic JSON serializable types
        if isinstance(self.data, (bool, int, str)):
            return self.data
            
        # Handle float with special cases for NaN/Inf
        if isinstance(self.data, float):
            if math.isnan(self.data) or math.isinf(self.data):
                # Instead of returning null, use a reasonable default value
                if math.isnan(self.data):
                    return 0.0
                elif math.isinf(self.data) and self.data > 0:
                    return 1.0e6  # Large positive value
                else:
                    return -1.0e6  # Large negative value
            return self.data
        
        # Handle complex numbers
        if isinstance(self.data, complex):
            real_part = self.data.real if not (math.isnan(self.data.real) or math.isinf(self.data.real)) else 0.0
            imag_part = self.data.imag if not (math.isnan(self.data.imag) or math.isinf(self.data.imag)) else 0.0
            return {
                "real": real_part,
                "imag": imag_part,
                "__complex__": True
            }
        
        # Handle NumPy arrays - this is where most issues occur
        if isinstance(self.data, np.ndarray):
            # Repair arrays instead of just filtering out bad values
            if self.aggressive_repair:
                # Handle complex arrays specially
                if np.issubdtype(self.data.dtype, np.complexfloating):
                    # Repair real and imaginary parts separately
                    real_part = sanitize_array(self.data.real, repair_mode='interpolate')
                    imag_part = sanitize_array(self.data.imag, repair_mode='interpolate')
                    # Recombine
                    sanitized_data = real_part + 1j * imag_part
                else:
                    # Regular array - repair directly
                    sanitized_data = sanitize_array(self.data, repair_mode='interpolate')
                    
                # If sanitization still leaves non-finite values, try more aggressive repair
                if np.any(~np.isfinite(sanitized_data)):
                    sanitized_data, _, _ = detect_and_fix_mhd_issues(sanitized_data, repair_mode='aggressive')
                    
                # Convert to list for JSON serialization
                return self.ensure_json_serializable(sanitized_data.tolist(), self.max_depth, self.current_depth + 1, self.aggressive_repair)
            else:
                # Original behavior - just convert to list
                return self.ensure_json_serializable(self.data.tolist(), self.max_depth, self.current_depth + 1, self.aggressive_repair)
        
        # Handle NumPy scalars
        if isinstance(self.data, np.number):
            if np.isnan(self.data) or np.isinf(self.data):
                if np.isnan(self.data):
                    return 0.0
                elif np.isposinf(self.data):
                    return 1.0e6
                else:
                    return -1.0e6
            return float(self.data)
        
        # Handle NumPy booleans
        if isinstance(self.data, np.bool_):
            return bool(self.data)
        
        # Handle lists and tuples
        if isinstance(self.data, (list, tuple)):
            return [self.ensure_json_serializable(item, self.max_depth, self.current_depth + 1, self.aggressive_repair) for item in self.data]
        
        # Handle sets
        if isinstance(self.data, set):
            return [self.ensure_json_serializable(item, self.max_depth, self.current_depth + 1, self.aggressive_repair) for item in self.data]
        
        # Handle dictionaries
        if isinstance(self.data, dict):
            result = {}
            for key, value in self.data.items():
                # Convert non-string keys to strings
                if not isinstance(key, str):
                    key = str(key)
                result[key] = self.ensure_json_serializable(value, self.max_depth, self.current_depth + 1, self.aggressive_repair)
            return result
        
        # Handle decimal.Decimal
        if isinstance(self.data, Decimal):
            return float(self.data)
        
        # Handle datetime objects
        if isinstance(self.data, (datetime.datetime, datetime.date)):
            try:
                return self.data.isoformat()
            except Exception as e:
                logger.warning(f"Failed to serialize datetime object: {e}")
                return str(self.data)
        
        # Handle other objects by trying to convert to string
        try:
            return str(self.data)
        except Exception as e:
            logger.error(f"Failed to serialize object of type {type(self.data)}: {e}")
            return f"<Unserializable object of type {type(self.data).__name__}>" 