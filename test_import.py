#!/usr/bin/env python
"""
Simple test script to verify imports are working correctly.
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    import myproject.utils.mhd.core as core
    print("✓ Successfully imported myproject.utils.mhd.core")
    
    # Test flatten_3d_array import
    from myproject.utils.numerical.tensor_utils import flatten_3d_array
    print("✓ Successfully imported flatten_3d_array")
    
    # Test create_grid import
    from myproject.utils.differential_operators import create_grid
    print("✓ Successfully imported create_grid")
    
    print("\nAll imports successful!")
except ImportError as e:
    print(f"Error importing modules: {e}")
    
    print("\nDebug information:")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # List the contents of the myproject directory to help debug
    try:
        print("\nContents of myproject directory:")
        for item in os.listdir(os.path.join(os.path.dirname(__file__), "myproject")):
            print(f"  - {item}")
    except Exception as list_error:
        print(f"Error listing directory: {list_error}") 