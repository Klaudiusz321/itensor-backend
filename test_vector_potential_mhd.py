#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for vector potential initialization in MHD.

This script defines functions to create and visualize magnetic fields from
different vector potentials, including their divergence.
"""
import os
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_magnetic_field(Bx, By, x, y, title="Magnetic Field"):
    """
    Plot a magnetic field using coordinates and field components.
    
    Args:
        Bx: x-component of magnetic field
        By: y-component of magnetic field
        x: x-coordinates grid
        y: y-coordinates grid
        title: Title for the plot
    """
    # Calculate the magnitude of the magnetic field
    B_mag = np.sqrt(Bx**2 + By**2)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot the magnitude as a contour plot
    contour = plt.contourf(x, y, B_mag, cmap='viridis')
    plt.colorbar(contour, label='|B|')
    
    # Add vector field visualization
    skip = 10  # Skip some points to make the plot clearer
    plt.quiver(x[::skip, ::skip], y[::skip, ::skip], 
               Bx[::skip, ::skip], By[::skip, ::skip], 
               angles='xy', scale_units='xy', scale=30, color='white')
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Save the plot
    output_filename = f"{title.replace(' ', '_')}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved magnetic field plot to {output_filename}")

def plot_divergence(div_B, x, y, title="Magnetic Field Divergence"):
    """
    Plot the divergence of a magnetic field.
    
    Args:
        div_B: Divergence of the magnetic field
        x: x-coordinates grid
        y: y-coordinates grid
        title: Title for the plot
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot the divergence as a contour plot with a diverging colormap
    # centered at zero
    vmax = np.max(np.abs(div_B))
    vmin = -vmax
    if vmax < 1e-14:  # If essentially zero, use a small range
        vmax = 1e-14
        vmin = -vmax
    
    contour = plt.contourf(x, y, div_B, cmap='RdBu_r', levels=50,
                           vmin=vmin, vmax=vmax)
    plt.colorbar(contour, label='∇·B')
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Save the plot
    output_filename = f"{title.replace(' ', '_')}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved divergence plot to {output_filename}")

def field_from_vector_potential(vector_potential_func, nx=100, ny=100, Lx=2.0, Ly=2.0):
    """
    Create a magnetic field from a vector potential function.
    
    Args:
        vector_potential_func: Function that takes (x, y) and returns the z-component
                              of the vector potential
        nx, ny: Number of grid points in x and y directions
        Lx, Ly: Domain size in x and y directions
        
    Returns:
        Tuple containing (Bx, By, div_B, x, y)
    """
    # Create coordinates
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the vector potential
    A_z = vector_potential_func(X, Y)
    
    # Calculate the magnetic field components
    # B_x = ∂A_z/∂y
    B_x = np.zeros_like(A_z)
    dy = Ly / (ny - 1)
    B_x[1:-1, :] = (A_z[2:, :] - A_z[:-2, :]) / (2 * dy)
    
    # B_y = -∂A_z/∂x
    B_y = np.zeros_like(A_z)
    dx = Lx / (nx - 1)
    B_y[:, 1:-1] = -(A_z[:, 2:] - A_z[:, :-2]) / (2 * dx)
    
    # Calculate divergence
    # div_B = ∂B_x/∂x + ∂B_y/∂y
    div_B = np.zeros_like(A_z)
    div_B[1:-1, 1:-1] = (B_x[1:-1, 2:] - B_x[1:-1, :-2]) / (2 * dx) + \
                        (B_y[2:, 1:-1] - B_y[:-2, 1:-1]) / (2 * dy)
    
    return B_x, B_y, div_B, X, Y

def create_sine_field(nx=100, ny=100, Lx=2.0, Ly=2.0):
    """Create a magnetic field from a sine vector potential."""
    def sine_vector_potential(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    return field_from_vector_potential(sine_vector_potential, nx, ny, Lx, Ly)

def create_dipole_field(nx=100, ny=100, Lx=2.0, Ly=2.0):
    """Create a magnetic field from a dipole vector potential."""
    def dipole_vector_potential(x, y):
        # Center of the dipole
        x0, y0 = Lx/2, Ly/2
        
        # Distance from center (add small epsilon to avoid singularity)
        r = np.sqrt((x - x0)**2 + (y - y0)**2) + 1e-6
        
        # Vector potential for a dipole
        return np.log(r)
    
    return field_from_vector_potential(dipole_vector_potential, nx, ny, Lx, Ly)

def create_magnetic_loop(nx=100, ny=100, Lx=2.0, Ly=2.0):
    """Create a magnetic field from a circular loop vector potential."""
    def loop_vector_potential(x, y):
        # Center of the loop
        x0, y0 = Lx/2, Ly/2
        
        # Distance from center
        r = np.sqrt((x - x0)**2 + (y - y0)**2)
        
        # Loop radius
        R = 0.3 * Lx
        
        # Vector potential: Az = max(0, R - r)^2
        # This creates a circular loop of magnetic field
        return np.maximum(0, R - r)**2
    
    return field_from_vector_potential(loop_vector_potential, nx, ny, Lx, Ly)

def create_quadrupole_field(nx=100, ny=100, Lx=2.0, Ly=2.0):
    """Create a magnetic field from a quadrupole vector potential."""
    def quadrupole_vector_potential(x, y):
        # Center of the quadrupole
        x0, y0 = Lx/2, Ly/2
        
        # Normalized coordinates
        x_norm = (x - x0) / (Lx/2)
        y_norm = (y - y0) / (Ly/2)
        
        # Vector potential for a quadrupole: Az = x^2 - y^2
        return x_norm**2 - y_norm**2
    
    return field_from_vector_potential(quadrupole_vector_potential, nx, ny, Lx, Ly)

def create_combined_plot(field_data_list, titles, output_filename="Combined_Magnetic_Fields.png"):
    """
    Create a combined plot showing multiple magnetic fields.
    
    Args:
        field_data_list: List of tuples (B_x, B_y, div_B, x, y)
        titles: List of titles for each field
        output_filename: Filename for the output plot
    """
    n_fields = len(field_data_list)
    fig, axes = plt.subplots(2, n_fields, figsize=(n_fields*5, 10))
    
    # For each field, plot both the magnetic field and its divergence
    for i, ((B_x, B_y, div_B, x, y), title) in enumerate(zip(field_data_list, titles)):
        # Calculate the magnitude of the magnetic field
        B_mag = np.sqrt(B_x**2 + B_y**2)
        
        # Plot the magnetic field in the top row
        contour = axes[0, i].contourf(x, y, B_mag, cmap='viridis')
        fig.colorbar(contour, ax=axes[0, i], label='|B|')
        
        # Add vector field visualization
        skip = 15  # Skip more points for the combined plot
        axes[0, i].quiver(x[::skip, ::skip], y[::skip, ::skip], 
                       B_x[::skip, ::skip], B_y[::skip, ::skip], 
                       angles='xy', scale_units='xy', scale=30, color='white')
        
        axes[0, i].set_title(title)
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('y')
        
        # Plot the divergence in the bottom row
        vmax = np.max(np.abs(div_B))
        vmin = -vmax
        if vmax < 1e-14:
            vmax = 1e-14
            vmin = -vmax
            
        contour_div = axes[1, i].contourf(x, y, div_B, cmap='RdBu_r', 
                                      levels=20, vmin=vmin, vmax=vmax)
        fig.colorbar(contour_div, ax=axes[1, i], label='∇·B')
        
        axes[1, i].set_title(f"{title} Divergence")
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved combined plot to {output_filename}")

def main():
    """
    Main function to demonstrate multiple magnetic fields from vector potentials.
    """
    resolution = 200  # Grid resolution
    
    try:
        # Store field data for combined plot
        field_data_list = []
        titles = []
        
        # Test magnetic field from sine wave vector potential
        logger.info("Creating sine wave magnetic field")
        sine_data = create_sine_field(nx=resolution, ny=resolution)
        B_x, B_y, div_B, x, y = sine_data
        field_data_list.append(sine_data)
        titles.append("Sine Wave")
        plot_magnetic_field(B_x, B_y, x, y, title="Sine Wave Magnetic Field")
        plot_divergence(div_B, x, y, title="Sine Wave Divergence")
        max_div = np.max(np.abs(div_B))
        logger.info(f"Sine wave maximum divergence: {max_div:.6e}")
        
        # Test magnetic field from dipole vector potential
        logger.info("Creating dipole magnetic field")
        dipole_data = create_dipole_field(nx=resolution, ny=resolution)
        B_x, B_y, div_B, x, y = dipole_data
        field_data_list.append(dipole_data)
        titles.append("Dipole")
        plot_magnetic_field(B_x, B_y, x, y, title="Dipole Magnetic Field")
        plot_divergence(div_B, x, y, title="Dipole Divergence")
        max_div = np.max(np.abs(div_B))
        logger.info(f"Dipole maximum divergence: {max_div:.6e}")
        
        # Test magnetic field from circular loop vector potential
        logger.info("Creating magnetic loop field")
        loop_data = create_magnetic_loop(nx=resolution, ny=resolution)
        B_x, B_y, div_B, x, y = loop_data
        field_data_list.append(loop_data)
        titles.append("Magnetic Loop")
        plot_magnetic_field(B_x, B_y, x, y, title="Magnetic Loop Field")
        plot_divergence(div_B, x, y, title="Magnetic Loop Divergence")
        max_div = np.max(np.abs(div_B))
        logger.info(f"Loop maximum divergence: {max_div:.6e}")
        
        # Test magnetic field from quadrupole vector potential
        logger.info("Creating quadrupole magnetic field")
        quadrupole_data = create_quadrupole_field(nx=resolution, ny=resolution)
        B_x, B_y, div_B, x, y = quadrupole_data
        field_data_list.append(quadrupole_data)
        titles.append("Quadrupole")
        plot_magnetic_field(B_x, B_y, x, y, title="Quadrupole Magnetic Field")
        plot_divergence(div_B, x, y, title="Quadrupole Divergence")
        max_div = np.max(np.abs(div_B))
        logger.info(f"Quadrupole maximum divergence: {max_div:.6e}")
        
        # Create combined plot
        logger.info("Creating combined visualization")
        create_combined_plot(field_data_list, titles)
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 