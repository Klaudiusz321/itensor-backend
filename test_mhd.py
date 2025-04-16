#!/usr/bin/env python
"""
Test script to verify that the MHD simulations can be initialized correctly.
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    # Import the MHD module
    from myproject.utils.mhd.core import magnetic_rotor_2d, orszag_tang_vortex_2d
    
    print("Testing MHD initialization...")
    
    # Try to initialize the magnetic rotor simulation
    domain_size = [(0.0, 1.0), (0.0, 1.0)]
    resolution = [64, 64]
    
    print("Initializing magnetic rotor simulation...")
    mhd_rotor = magnetic_rotor_2d(domain_size, resolution)
    print("✓ Magnetic rotor initialization successful")
    
    print("Initializing Orszag-Tang vortex simulation...")
    mhd_orszag = orszag_tang_vortex_2d(domain_size, resolution)
    print("✓ Orszag-Tang vortex initialization successful")
    
    print("\nAll MHD simulations initialized successfully!")
    
except Exception as e:
    print(f"Error initializing MHD simulations: {e}")
    import traceback
    traceback.print_exc() 