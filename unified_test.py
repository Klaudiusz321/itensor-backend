#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified MHD Simulation Visualization

This script demonstrates a complete MHD simulation with real-time visualization.
It runs a magnetic rotor test case and displays the evolving magnetic field and its
divergence using Matplotlib's animation capabilities.

Key features:
- Initializes an MHD system with magnetic rotor initial conditions
- Runs the simulation while dynamically visualizing the results
- Displays diagnostic information in real-time
- Shows both the magnetic field and its divergence
- Maintains the divergence-free condition through constrained transport
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    # Import required modules
    from myproject.utils.mhd.core import MHDSystem, magnetic_rotor_2d
    from myproject.utils.differential_operators.numeric import evaluate_divergence
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running this script from the right directory.")
    sys.exit(1)

class MHDSimulation:
    """Class to manage the MHD simulation and visualization."""
    
    def __init__(self, domain_size=[(0.0, 1.0), (0.0, 1.0)], resolution=[64, 64], 
                 final_time=0.5, frames=50):
        """
        Initialize the MHD simulation.
        
        Args:
            domain_size: Size of the computational domain [(xmin, xmax), (ymin, ymax)]
            resolution: Grid resolution [nx, ny]
            final_time: Total simulation time
            frames: Number of animation frames to produce
        """
        self.domain_size = domain_size
        self.resolution = resolution
        self.final_time = final_time
        self.frames = frames
        
        # Simulation state
        self.mhd_system = None
        self.current_time = 0.0
        self.max_div_B = 0.0
        self.time_step = 0.0
        
        # Data storage for animation
        self.time_history = []
        self.div_history = []
        self.energy_history = []
        
        # Animation components
        self.fig = None
        self.axes = None
        self.plots = {}
        self.animation = None
        
        logger.info("MHD Simulation initialized")
    
    def initialize_simulation(self):
        """Initialize the MHD system with magnetic rotor initial conditions."""
        try:
            logger.info("Creating magnetic rotor simulation...")
            self.mhd_system = magnetic_rotor_2d(
                domain_size=self.domain_size,
                resolution=self.resolution
            )
            
            # Compute initial time step
            self.time_step = self.mhd_system.compute_time_step()
            
            # Check initial divergence
            self.max_div_B = self.mhd_system.check_divergence_free()
            
            # Store initial state
            self.current_time = 0.0
            self.time_history.append(self.current_time)
            self.div_history.append(self.max_div_B)
            
            # Calculate total energy
            total_energy = np.sum(self.mhd_system.conserved_vars['energy'])
            self.energy_history.append(total_energy)
            
            logger.info(f"Simulation initialized - dt: {self.time_step:.5f}, max |∇·B|: {self.max_div_B:.2e}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize simulation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def advance_simulation(self):
        """Advance the simulation by one time step."""
        try:
            # Update time step if needed
            self.time_step = self.mhd_system.compute_time_step()
            
            # Advance the simulation by one time step
            self.mhd_system.advance_time_step()
            
            # Update current time
            self.current_time = self.mhd_system.time
            
            # Calculate max divergence
            self.max_div_B = self.mhd_system.check_divergence_free()
            
            # Store history
            self.time_history.append(self.current_time)
            self.div_history.append(self.max_div_B)
            
            # Calculate total energy
            total_energy = np.sum(self.mhd_system.conserved_vars['energy'])
            self.energy_history.append(total_energy)
            
            return True
            
        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def setup_visualization(self):
        """Set up the visualization figure and axes."""
        # Create figure with multiple subplots
        self.fig = plt.figure(figsize=(14, 8))
        grid = plt.GridSpec(2, 3, wspace=0.3, hspace=0.3)
        
        # Create axes for different plots
        self.axes = {
            'magnetic_field': self.fig.add_subplot(grid[0, 0:2]),
            'velocity_field': self.fig.add_subplot(grid[1, 0:2]),
            'divergence': self.fig.add_subplot(grid[0, 2]),
            'diagnostics': self.fig.add_subplot(grid[1, 2])
        }
        
        # Set titles
        self.axes['magnetic_field'].set_title('Magnetic Field')
        self.axes['velocity_field'].set_title('Velocity Field')
        self.axes['divergence'].set_title('Magnetic Divergence')
        self.axes['diagnostics'].set_title('Diagnostics')
        
        # Initialize plots
        x, y = self.mhd_system.grid
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Magnetic field magnitude
        Bx, By = self.mhd_system.magnetic_field[0], self.mhd_system.magnetic_field[1]
        B_mag = np.sqrt(Bx**2 + By**2)
        
        # Create magnetic field plot with streamplot and magnitude
        self.plots['B_mag'] = self.axes['magnetic_field'].imshow(
            B_mag.T, origin='lower', 
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap='viridis', 
            interpolation='nearest'
        )
        self.plots['B_stream'] = self.axes['magnetic_field'].streamplot(
            X, Y, Bx, By, color='white', density=1
        )
        plt.colorbar(self.plots['B_mag'], ax=self.axes['magnetic_field'], label='|B|')
        
        # Velocity field
        Vx, Vy = self.mhd_system.velocity[0], self.mhd_system.velocity[1]
        V_mag = np.sqrt(Vx**2 + Vy**2)
        
        self.plots['V_mag'] = self.axes['velocity_field'].imshow(
            V_mag.T, origin='lower', 
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap='plasma', 
            interpolation='nearest'
        )
        self.plots['V_stream'] = self.axes['velocity_field'].streamplot(
            X, Y, Vx, Vy, color='white', density=1
        )
        plt.colorbar(self.plots['V_mag'], ax=self.axes['velocity_field'], label='|V|')
        
        # Divergence plot
        div_B = self.compute_divergence()
        vmax = max(np.max(np.abs(div_B)), 1e-10)
        self.plots['div_B'] = self.axes['divergence'].imshow(
            div_B.T, origin='lower', 
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap='RdBu_r', vmin=-vmax, vmax=vmax,
            interpolation='nearest'
        )
        plt.colorbar(self.plots['div_B'], ax=self.axes['divergence'], label='∇·B')
        
        # Diagnostics panel - initially empty
        self.axes['diagnostics'].set_xlim(0, self.final_time)
        self.axes['diagnostics'].set_xlabel('Time')
        
        # Create divergence history plot
        self.plots['div_history'], = self.axes['diagnostics'].semilogy(
            self.time_history, self.div_history, 'r-', label='Max |∇·B|'
        )
        
        # Create energy history plot (second y-axis)
        self.axes_energy = self.axes['diagnostics'].twinx()
        self.plots['energy_history'], = self.axes_energy.plot(
            self.time_history, [e/self.energy_history[0] for e in self.energy_history], 
            'b-', label='Rel. Energy'
        )
        self.axes_energy.set_ylim(0.95, 1.05)
        
        # Add legends
        lines1, labels1 = self.axes['diagnostics'].get_legend_handles_labels()
        lines2, labels2 = self.axes_energy.get_legend_handles_labels()
        self.axes['diagnostics'].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Add simulation time as text
        self.plots['time_text'] = self.fig.text(
            0.5, 0.01, f'Time: {self.current_time:.3f}', 
            ha='center', va='bottom', fontsize=12
        )
        
        # Add a suptitle to the figure
        self.fig.suptitle('Magnetic Rotor MHD Simulation', fontsize=16)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for the time text at bottom
        
        return True
    
    def compute_divergence(self):
        """Compute the divergence of the magnetic field."""
        return evaluate_divergence(self.mhd_system.magnetic_field, 
                                   np.eye(len(self.resolution)), 
                                   self.mhd_system.grid)
    
    def update_visualization(self, frame):
        """Update the visualization for a given animation frame."""
        try:
            # For the first frame, we don't need to advance the simulation
            if frame > 0:
                # Calculate time step for this frame
                dt_per_frame = self.final_time / self.frames
                
                # Run simulation until we reach the next frame time
                target_time = frame * dt_per_frame
                
                while self.current_time < target_time:
                    success = self.advance_simulation()
                    if not success:
                        logger.error("Failed to advance simulation, stopping animation")
                        self.animation.event_source.stop()
                        return
            
            # Update magnetic field plot
            Bx, By = self.mhd_system.magnetic_field[0], self.mhd_system.magnetic_field[1]
            B_mag = np.sqrt(Bx**2 + By**2)
            self.plots['B_mag'].set_array(B_mag.T)
            
            # Need to completely recreate the streamplot
            self.axes['magnetic_field'].collections = []
            self.axes['magnetic_field'].patches = []
            x, y = self.mhd_system.grid
            X, Y = np.meshgrid(x, y, indexing='ij')
            self.axes['magnetic_field'].streamplot(
                X, Y, Bx, By, color='white', density=1
            )
            
            # Update velocity field plot
            Vx, Vy = self.mhd_system.velocity[0], self.mhd_system.velocity[1]
            V_mag = np.sqrt(Vx**2 + Vy**2)
            self.plots['V_mag'].set_array(V_mag.T)
            
            # Recreate velocity streamplot
            self.axes['velocity_field'].collections = []
            self.axes['velocity_field'].patches = []
            self.axes['velocity_field'].streamplot(
                X, Y, Vx, Vy, color='white', density=1
            )
            
            # Update divergence plot
            div_B = self.compute_divergence()
            vmax = max(np.max(np.abs(div_B)), 1e-10)
            self.plots['div_B'].set_array(div_B.T)
            self.plots['div_B'].set_clim(-vmax, vmax)
            
            # Update diagnostic plots
            self.plots['div_history'].set_data(self.time_history, self.div_history)
            self.plots['energy_history'].set_data(
                self.time_history, [e/self.energy_history[0] for e in self.energy_history]
            )
            
            # Update time text
            self.plots['time_text'].set_text(f'Time: {self.current_time:.3f}, Max |∇·B|: {self.max_div_B:.2e}')
            
            # Adjust y-limits for divergence plot if needed
            max_div = max(self.div_history)
            if max_div > 0:
                self.axes['diagnostics'].set_ylim(max(1e-16, max_div/1e3), max(1e-8, max_div*10))
            
            logger.info(f"Visualization updated - frame: {frame}, time: {self.current_time:.3f}")
            
            # Return all the artists that need to be redrawn
            return list(self.plots.values())
            
        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def run_animation(self):
        """Run the animated simulation."""
        try:
            logger.info("Starting animation...")
            
            # Create the animation
            self.animation = FuncAnimation(
                self.fig, self.update_visualization,
                frames=self.frames,
                interval=100,  # milliseconds between frames
                blit=False,    # redraw the entire figure for simplicity
                repeat=False   # don't repeat the animation
            )
            
            # Display the animation
            plt.show()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in animation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_animation(self, filename='mhd_simulation.mp4'):
        """Save the animation to a file."""
        try:
            logger.info(f"Saving animation to {filename}...")
            self.animation.save(filename, writer='ffmpeg', fps=10, dpi=200)
            logger.info(f"Animation saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save animation: {e}")
            return False

def main():
    """Main function to run the MHD simulation with visualization."""
    try:
        # Create and initialize the simulation
        simulation = MHDSimulation(
            domain_size=[(0.0, 1.0), (0.0, 1.0)],
            resolution=[100, 100],  # Higher resolution for better visualization
            final_time=0.4,
            frames=40
        )
        
        # Initialize the MHD system
        if not simulation.initialize_simulation():
            logger.error("Failed to initialize simulation, exiting")
            return False
        
        # Setup visualization
        if not simulation.setup_visualization():
            logger.error("Failed to setup visualization, exiting")
            return False
        
        # Run the animation
        if not simulation.run_animation():
            logger.error("Animation failed, exiting")
            return False
        
        # Optionally save the animation (uncomment to enable)
        # simulation.save_animation()
        
        logger.info("Simulation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
