import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from myproject.utils.mhd.core import orszag_tang_vortex_2d
    
    # Create MHD system
    logger.info("Creating MHD system (Orszag-Tang vortex)")
    mhd = orszag_tang_vortex_2d([(0.0, 1.0), (0.0, 1.0)], [64, 64])
    
    # Number of steps to run
    num_steps = 5
    
    # Initial state values for tracking changes
    initial_density = mhd.density.copy()
    initial_magnetic = [b.copy() for b in mhd.magnetic_field]
    
    # Run evolution for a few steps
    for step in range(num_steps):
        logger.info(f"Step {step+1}/{num_steps}")
        
        # Advance one time step
        mhd.advance_time_step()
        
        # Report statistics
        density_change = np.max(np.abs(mhd.density - initial_density))
        B_change = np.max([np.max(np.abs(mhd.magnetic_field[i] - initial_magnetic[i])) 
                           for i in range(len(mhd.magnetic_field))])
        
        # Get max/min values for key fields
        density_min, density_max = np.min(mhd.density), np.max(mhd.density)
        pressure_min, pressure_max = np.min(mhd.pressure), np.max(mhd.pressure)
        
        logger.info(f"Time: {mhd.time:.6f}, dt: {mhd.dt:.6f}")
        logger.info(f"Density change: {density_change:.6e}")
        logger.info(f"Magnetic field change: {B_change:.6e}")
        logger.info(f"Density min/max: {density_min:.6f}/{density_max:.6f}")
        logger.info(f"Pressure min/max: {pressure_min:.6f}/{pressure_max:.6f}")
    
    # Save a simple plot of the final density
    plt.figure(figsize=(8, 6))
    plt.imshow(mhd.density.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    plt.colorbar(label='Density')
    plt.title(f'Density at time {mhd.time:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('density.png')
    plt.close()
    
    # Plot magnetic field magnitude
    B_mag = np.sqrt(mhd.magnetic_field[0]**2 + mhd.magnetic_field[1]**2)
    plt.figure(figsize=(8, 6))
    plt.imshow(B_mag.T, origin='lower', extent=[0, 1, 0, 1], cmap='magma')
    plt.colorbar(label='Magnetic Field Magnitude')
    plt.title(f'Magnetic Field at time {mhd.time:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('magnetic_field.png')
    plt.close()
    
    logger.info("Simulation completed successfully")
    
except Exception as e:
    logger.error(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1) 