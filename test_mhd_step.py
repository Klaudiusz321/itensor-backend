import numpy as np
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
    
    # Check initial state
    logger.info(f"Initial density shape: {mhd.density.shape}")
    logger.info(f"Initial velocity shapes: {[v.shape for v in mhd.velocity]}")
    logger.info(f"Initial magnetic field shapes: {[b.shape for b in mhd.magnetic_field]}")
    logger.info(f"Initial face-centered B shapes: {[b.shape for b in mhd.face_centered_b]}")
    
    # Check conserved_vars
    logger.info(f"Conserved vars keys: {mhd.conserved_vars.keys()}")
    for key, value in mhd.conserved_vars.items():
        if isinstance(value, list):
            logger.info(f"Conserved {key} shapes: {[v.shape for v in value]}")
        else:
            logger.info(f"Conserved {key} shape: {value.shape}")
    
    # Check time step
    logger.info("Computing time step")
    dt = mhd.compute_time_step()
    logger.info(f"Time step: {dt}")
    
    # Run a single step
    logger.info("Advancing time step")
    mhd.advance_time_step()
    logger.info("Step completed successfully")
    
    # Check divergence
    div_b = mhd.check_divergence_free()
    logger.info(f"Max |div(B)| after step: {div_b:.6e}")
    
except Exception as e:
    logger.error(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1) 