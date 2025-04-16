"""
Magnetic Rotor Demo Script.

This script demonstrates the MHD implementation by running the magnetic
rotor simulation and visualizing the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
import time
import os
import sys

# Add the project root to the path if running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import MHD modules
from myproject.utils.mhd.core import MHDSystem
from myproject.utils.mhd.initial_conditions import magnetic_rotor

def magnetic_rotor_demo(resolution=(256, 256), final_time=0.4, output_interval=0.05, plot=True):
    """
    Run the magnetic rotor simulation and visualize the results.
    
    The magnetic rotor problem involves a dense rotating disk in a uniform
    magnetic field, generating torsional Alfvén waves.
    
    Args:
        resolution: Grid resolution (nx, ny)
        final_time: Time to evolve the system to
        output_interval: Time interval between outputs
        plot: Whether to create plots during the simulation
        
    Returns:
        Dictionary with final simulation state
    """
    print("Setting up magnetic rotor simulation...")
    
    # Define domain and coordinate system
    domain_size = [(0.0, 1.0), (0.0, 1.0)]  # 2D domain [0,1] x [0,1]
    coordinate_system = {
        'name': 'cartesian',
        'coordinates': ['x', 'y'],
        'transformation': None
    }
    
    # Create MHD system with constrained transport for div(B) = 0
    mhd = MHDSystem(coordinate_system, domain_size, resolution, 
                   gamma=5/3, use_constrained_transport=True)
    
    # Initialize with magnetic rotor problem
    rotor_init = magnetic_rotor(mhd.grid, gamma=5/3)
    
    # Set initial conditions
    mhd.set_initial_conditions(
        lambda x, y: rotor_init['density'],
        [lambda x, y: rotor_init['velocity'][0], lambda x, y: rotor_init['velocity'][1]],
        lambda x, y: rotor_init['pressure'],
        [lambda x, y: rotor_init['magnetic_field'][0], lambda x, y: rotor_init['magnetic_field'][1]]
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
        
        # Print status and div(B) check
        div_b = mhd_system.check_divergence_free()
        print(f"Time: {mhd_system.time:.4f}, dt: {mhd_system.dt:.6f}, max |div(B)|: {div_b:.2e}")
        
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
    mach_number = v_mag / np.sqrt(mhd_system.gamma * p / rho)
    alfven_mach = v_mag / (B_mag / np.sqrt(rho + 1e-10))  # Add small number to avoid division by zero
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Magnetic Rotor at t = {mhd_system.time:.3f}", fontsize=16)
    
    # Plot density with streamlines
    im1 = axs[0, 0].imshow(rho.T, origin='lower', cmap='viridis')
    axs[0, 0].set_title('Density')
    fig.colorbar(im1, ax=axs[0, 0])
    
    # Downsample for streamlines (to avoid cluttering)
    ds = 4
    y, x = np.mgrid[0:rho.shape[0]:ds, 0:rho.shape[1]:ds]
    axs[0, 0].streamplot(x, y, vx[::ds, ::ds].T, vy[::ds, ::ds].T, 
                        color='white', linewidth=0.5, density=1.5)
    
    # Plot magnetic field magnitude with field lines
    im2 = axs[0, 1].imshow(B_mag.T, origin='lower', cmap='magma')
    axs[0, 1].set_title('Magnetic Field Magnitude')
    fig.colorbar(im2, ax=axs[0, 1])
    
    axs[0, 1].streamplot(x, y, Bx[::ds, ::ds].T, By[::ds, ::ds].T, 
                        color='white', linewidth=0.5, density=1.5)
    
    # Plot pressure
    im3 = axs[0, 2].imshow(p.T, origin='lower', cmap='plasma')
    axs[0, 2].set_title('Pressure')
    fig.colorbar(im3, ax=axs[0, 2])
    
    # Plot velocity magnitude
    im4 = axs[1, 0].imshow(v_mag.T, origin='lower', cmap='cividis')
    axs[1, 0].set_title('Velocity Magnitude')
    fig.colorbar(im4, ax=axs[1, 0])
    
    # Plot Mach number
    im5 = axs[1, 1].imshow(mach_number.T, origin='lower', cmap='inferno', 
                         norm=LogNorm(vmin=0.01, vmax=max(1.0, np.max(mach_number))))
    axs[1, 1].set_title('Mach Number')
    fig.colorbar(im5, ax=axs[1, 1])
    
    # Plot Alfvén Mach number
    im6 = axs[1, 2].imshow(alfven_mach.T, origin='lower', cmap='viridis', 
                         norm=LogNorm(vmin=0.01, vmax=max(1.0, np.max(alfven_mach))))
    axs[1, 2].set_title('Alfvén Mach Number')
    fig.colorbar(im6, ax=axs[1, 2])
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('output', exist_ok=True)
    plt.savefig(f'output/magnetic_rotor_frame_{frame_idx:04d}.png', dpi=150)
    plt.close()

def create_animation(output_states, output_times):
    """
    Create an animation from the sequence of output states.
    
    Args:
        output_states: List of output states
        output_times: List of output times
    """
    print("Creating animation...")
    
    # Set up the figure and axis for density
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    im1 = ax1.imshow(output_states[0]['density'].T, origin='lower', cmap='viridis')
    title1 = ax1.set_title('Magnetic Rotor: Density at t = 0.000')
    fig1.colorbar(im1, ax=ax1, label='Density')
    
    # Update function for density animation
    def update_density(frame):
        im1.set_array(output_states[frame]['density'].T)
        title1.set_text(f'Magnetic Rotor: Density at t = {output_times[frame]:.3f}')
        return [im1, title1]
    
    # Create and save density animation
    anim1 = FuncAnimation(fig1, update_density, frames=len(output_states), interval=200, blit=True)
    os.makedirs('output', exist_ok=True)
    anim1.save('output/magnetic_rotor_density_animation.mp4', writer='ffmpeg', dpi=150)
    plt.close(fig1)
    
    # Set up the figure and axis for magnetic field
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    # Compute magnetic field magnitude for each frame
    B_mags = [np.sqrt(s['magnetic_field'][0]**2 + s['magnetic_field'][1]**2) for s in output_states]
    im2 = ax2.imshow(B_mags[0].T, origin='lower', cmap='magma')
    title2 = ax2.set_title('Magnetic Rotor: |B| at t = 0.000')
    fig2.colorbar(im2, ax=ax2, label='Magnetic Field Magnitude')
    
    # Update function for magnetic field animation
    def update_magnetic(frame):
        im2.set_array(B_mags[frame].T)
        title2.set_text(f'Magnetic Rotor: |B| at t = {output_times[frame]:.3f}')
        return [im2, title2]
    
    # Create and save magnetic field animation
    anim2 = FuncAnimation(fig2, update_magnetic, frames=len(output_states), interval=200, blit=True)
    anim2.save('output/magnetic_rotor_magnetic_animation.mp4', writer='ffmpeg', dpi=150)
    plt.close(fig2)

if __name__ == '__main__':
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run magnetic rotor simulation')
    parser.add_argument('--resolution', type=int, default=256, help='Grid resolution (default: 256)')
    parser.add_argument('--time', type=float, default=0.4, help='Final simulation time (default: 0.4)')
    parser.add_argument('--interval', type=float, default=0.05, help='Output interval (default: 0.05)')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    # Run the demo
    result = magnetic_rotor_demo(
        resolution=(args.resolution, args.resolution),
        final_time=args.time,
        output_interval=args.interval,
        plot=not args.no_plot
    )
    
    print("Demo completed successfully!")
    print(f"Final simulation time: {result['output_times'][-1]:.4f}")
    print(f"Number of output frames: {len(result['output_times'])}") 