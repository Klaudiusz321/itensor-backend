"""
Orszag-Tang Vortex Demo Script.

This script demonstrates the MHD implementation by running the Orszag-Tang
vortex simulation and visualizing the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import os
import sys

# Add the project root to the path if running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import MHD modules
from myproject.utils.mhd.core import MHDSystem
from myproject.utils.mhd.initial_conditions import orszag_tang_vortex
from myproject.utils.differential_operators import create_grid

def orszag_tang_demo(resolution=(128, 128), final_time=1.0, output_interval=0.1, plot=True):
    """
    Run the Orszag-Tang vortex simulation and visualize the results.
    
    Args:
        resolution: Grid resolution (nx, ny)
        final_time: Time to evolve the system to
        output_interval: Time interval between outputs
        plot: Whether to create plots during the simulation
        
    Returns:
        Dictionary with final simulation state
    """
    print("Setting up Orszag-Tang vortex simulation...")
    
    # Define domain and coordinate system
    domain_size = [(0.0, 1.0), (0.0, 1.0)]  # 2D domain [0,1] x [0,1]
    coordinate_system = {
        'name': 'cartesian',
        'coordinates': ['x', 'y'],
        'transformation': None
    }
    
    # Create MHD system
    mhd = MHDSystem(coordinate_system, domain_size, resolution, gamma=5/3)
    
    # Initialize with Orszag-Tang vortex
    ot_init = orszag_tang_vortex(mhd.grid, gamma=5/3)
    
    # Set initial conditions
    mhd.set_initial_conditions(
        lambda x, y: ot_init['density'],
        [lambda x, y: ot_init['velocity'][0], lambda x, y: ot_init['velocity'][1]],
        lambda x, y: ot_init['pressure'],
        [lambda x, y: ot_init['magnetic_field'][0], lambda x, y: ot_init['magnetic_field'][1]]
    )
    
    # Create a list to store output states for visualization
    output_states = []
    output_times = []
    
    # Output callback function to collect results
    def save_output(mhd_system):
        output_states.append({
            'density': np.copy(mhd_system.density),
            'velocity': [np.copy(v) for v in mhd_system.velocity],
            'pressure': np.copy(mhd_system.pressure),
            'magnetic_field': [np.copy(b) for b in mhd_system.magnetic_field],
        })
        output_times.append(mhd_system.time)
        
        # Print status
        print(f"Time: {mhd_system.time:.4f}, dt: {mhd_system.dt:.6f}, max speed: {mhd_system.max_wavespeed:.4f}")
        
        # Create plots if requested
        if plot:
            plot_state(mhd_system, len(output_times) - 1)
    
    # Run the simulation
    print(f"Evolving to time {final_time} with output interval {output_interval}...")
    start_time = time.time()
    
    # Save initial state
    save_output(mhd)
    
    # Evolve the system
    result = mhd.evolve(final_time, save_output, output_interval)
    
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Create animation if plot is requested
    if plot:
        create_animation(output_states, output_times)
    
    return {
        'final_state': result,
        'output_states': output_states,
        'output_times': output_times
    }

def plot_state(mhd_system, frame_idx):
    """
    Create plots of the current MHD state.
    
    Args:
        mhd_system: MHD system to plot
        frame_idx: Index of the current frame
    """
    # Extract data from the MHD system
    rho = mhd_system.density
    vx, vy = mhd_system.velocity
    p = mhd_system.pressure
    Bx, By = mhd_system.magnetic_field
    
    # Compute derived quantities
    v_mag = np.sqrt(vx**2 + vy**2)
    B_mag = np.sqrt(Bx**2 + By**2)
    
    # Compute kinetic and magnetic energy densities
    kin_energy = 0.5 * rho * v_mag**2
    mag_energy = 0.5 * B_mag**2
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Orszag-Tang Vortex at t = {mhd_system.time:.3f}", fontsize=16)
    
    # Plot density
    im1 = axs[0, 0].imshow(rho.T, origin='lower', cmap='viridis')
    axs[0, 0].set_title('Density')
    fig.colorbar(im1, ax=axs[0, 0])
    
    # Plot pressure
    im2 = axs[0, 1].imshow(p.T, origin='lower', cmap='plasma')
    axs[0, 1].set_title('Pressure')
    fig.colorbar(im2, ax=axs[0, 1])
    
    # Plot velocity magnitude
    im3 = axs[0, 2].imshow(v_mag.T, origin='lower', cmap='cividis')
    axs[0, 2].set_title('Velocity Magnitude')
    fig.colorbar(im3, ax=axs[0, 2])
    
    # Plot magnetic field magnitude
    im4 = axs[1, 0].imshow(B_mag.T, origin='lower', cmap='magma')
    axs[1, 0].set_title('Magnetic Field Magnitude')
    fig.colorbar(im4, ax=axs[1, 0])
    
    # Plot kinetic energy density
    im5 = axs[1, 1].imshow(kin_energy.T, origin='lower', cmap='inferno')
    axs[1, 1].set_title('Kinetic Energy Density')
    fig.colorbar(im5, ax=axs[1, 1])
    
    # Plot magnetic energy density
    im6 = axs[1, 2].imshow(mag_energy.T, origin='lower', cmap='viridis')
    axs[1, 2].set_title('Magnetic Energy Density')
    fig.colorbar(im6, ax=axs[1, 2])
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('output', exist_ok=True)
    plt.savefig(f'output/orszag_tang_frame_{frame_idx:04d}.png', dpi=150)
    plt.close()

def create_animation(output_states, output_times):
    """
    Create an animation from the sequence of output states.
    
    Args:
        output_states: List of output states
        output_times: List of output times
    """
    print("Creating animation...")
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Initialize plot
    im = ax.imshow(output_states[0]['density'].T, origin='lower', cmap='viridis', 
                   vmin=np.min([s['density'] for s in output_states]),
                   vmax=np.max([s['density'] for s in output_states]))
    
    title = ax.set_title('Orszag-Tang Vortex: Density at t = 0.000')
    fig.colorbar(im, ax=ax, label='Density')
    
    # Update function for animation
    def update(frame):
        im.set_array(output_states[frame]['density'].T)
        title.set_text(f'Orszag-Tang Vortex: Density at t = {output_times[frame]:.3f}')
        return [im, title]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(output_states), interval=200, blit=True)
    
    # Save animation
    os.makedirs('output', exist_ok=True)
    anim.save('output/orszag_tang_animation.mp4', writer='ffmpeg', dpi=150)
    
    plt.close()

if __name__ == '__main__':
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run Orszag-Tang vortex simulation')
    parser.add_argument('--resolution', type=int, default=128, help='Grid resolution (default: 128)')
    parser.add_argument('--time', type=float, default=1.0, help='Final simulation time (default: 1.0)')
    parser.add_argument('--interval', type=float, default=0.1, help='Output interval (default: 0.1)')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    # Run the demo
    result = orszag_tang_demo(
        resolution=(args.resolution, args.resolution),
        final_time=args.time,
        output_interval=args.interval,
        plot=not args.no_plot
    )
    
    print("Demo completed successfully!")
    print(f"Final simulation time: {result['output_times'][-1]:.4f}")
    print(f"Number of output frames: {len(result['output_times'])}") 