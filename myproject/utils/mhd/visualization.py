import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colormaps

def visualize_2d_scalar_field(mhd_system, field_name, time=None, ax=None, 
                            cmap='viridis', log_scale=False, vmin=None, 
                            vmax=None, colorbar=True, title=None):
    """
    Visualize a 2D scalar field from an MHD simulation.
    
    Args:
        mhd_system: MHD system containing the field data
        field_name: Name of the field to visualize
        time: Simulation time (for title)
        ax: Matplotlib axis for plotting (creates new figure if None)
        cmap: Colormap name
        log_scale: Whether to use logarithmic scale
        vmin, vmax: Minimum and maximum values for colormap
        colorbar: Whether to add a colorbar
        title: Custom title (default: field name)
    
    Returns:
        fig, ax: The figure and axis objects
    """
    # Extract grid coordinates (now from tuple)
    x, y = mhd_system.grid
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Get field data based on name
    if field_name == 'density':
        data = mhd_system.density
    elif field_name == 'pressure':
        data = mhd_system.pressure
    elif field_name == 'velocity_magnitude':
        data = np.sqrt(mhd_system.velocity[0]**2 + mhd_system.velocity[1]**2)
    elif field_name == 'magnetic_magnitude':
        data = np.sqrt(mhd_system.magnetic_field[0]**2 + mhd_system.magnetic_field[1]**2)
    elif field_name == 'vorticity':
        # Compute vorticity (curl of velocity in 2D: ∂vy/∂x - ∂vx/∂y)
        vx_dy = np.gradient(mhd_system.velocity[0], y, axis=1)
        vy_dx = np.gradient(mhd_system.velocity[1], x, axis=0)
        data = vy_dx - vx_dy
    elif field_name == 'current_density':
        # Compute current density (curl of B in 2D: ∂By/∂x - ∂Bx/∂y)
        bx_dy = np.gradient(mhd_system.magnetic_field[0], y, axis=1)
        by_dx = np.gradient(mhd_system.magnetic_field[1], x, axis=0)
        data = by_dx - bx_dy
    else:
        raise ValueError(f"Unknown field name: {field_name}")
    
    # Create new figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Define the plot
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
    im = ax.pcolormesh(X, Y, data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    
    # Add colorbar if requested
    if colorbar:
        plt.colorbar(im, ax=ax)
    
    # Set plot title
    if title is None:
        title = field_name.replace('_', ' ').title()
    if time is not None:
        title += f" at t = {time:.3f}"
    ax.set_title(title)
    
    # Set axis labels
    ax.set_xlabel(mhd_system.coord_names[0])
    ax.set_ylabel(mhd_system.coord_names[1])
    
    return fig, ax

def visualize_2d_vector_field(mhd_system, field_type='velocity', time=None, 
                             ax=None, density=1.0, scale=1.0, color='k',
                             background=None, bg_cmap='viridis', log_scale=False,
                             vmin=None, vmax=None, colorbar=True, title=None):
    """
    Visualize a 2D vector field from an MHD simulation.
    
    Args:
        mhd_system: MHD system containing the field data
        field_type: 'velocity' or 'magnetic'
        time: Simulation time (for title)
        ax: Matplotlib axis for plotting (creates new figure if None)
        density: Density of arrows in quiver plot
        scale: Scaling factor for arrows
        color: Color of arrows
        background: Background scalar field to display ('density', 'pressure', etc.)
        bg_cmap: Colormap for background field
        log_scale: Whether to use logarithmic scale for background
        vmin, vmax: Minimum and maximum values for background colormap
        colorbar: Whether to add a colorbar for background
        title: Custom title
    
    Returns:
        fig, ax: The figure and axis objects
    """
    # Extract grid coordinates (now from tuple)
    x, y = mhd_system.grid
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Get vector field components based on type
    if field_type == 'velocity':
        u = mhd_system.velocity[0]
        v = mhd_system.velocity[1]
        field_name = 'Velocity'
    elif field_type == 'magnetic':
        u = mhd_system.magnetic_field[0]
        v = mhd_system.magnetic_field[1]
        field_name = 'Magnetic Field'
    else:
        raise ValueError(f"Unknown field type: {field_type}")
    
    # Create new figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Plot background field if specified
    if background is not None:
        bg_fig, _ = visualize_2d_scalar_field(
            mhd_system, background, time=None, ax=ax, 
            cmap=bg_cmap, log_scale=log_scale, vmin=vmin, 
            vmax=vmax, colorbar=colorbar, title=None
        )
    
    # Subsample points for quiver plot based on density
    step = max(1, int(1.0 / density))
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    u_sub = u[::step, ::step]
    v_sub = v[::step, ::step]
    
    # Plot vector field
    ax.quiver(X_sub, Y_sub, u_sub, v_sub, color=color, scale_units='xy', 
             scale=scale, width=0.002)
    
    # Set plot title
    if title is None:
        title = field_name
    if time is not None:
        title += f" at t = {time:.3f}"
    ax.set_title(title)
    
    # Set axis labels
    ax.set_xlabel(mhd_system.coord_names[0])
    ax.set_ylabel(mhd_system.coord_names[1])
    
    return fig, ax

def visualize_2d_streamlines(mhd_system, field_type='velocity', time=None, 
                           ax=None, density=1.0, color='k', linewidth=1.0,
                           background=None, bg_cmap='viridis', log_scale=False,
                           vmin=None, vmax=None, colorbar=True, title=None):
    """
    Visualize streamlines of a 2D vector field from an MHD simulation.
    
    Args:
        mhd_system: MHD system containing the field data
        field_type: 'velocity' or 'magnetic'
        time: Simulation time (for title)
        ax: Matplotlib axis for plotting (creates new figure if None)
        density: Density of streamlines
        color: Color of streamlines
        linewidth: Width of streamlines
        background: Background scalar field to display ('density', 'pressure', etc.)
        bg_cmap: Colormap for background field
        log_scale: Whether to use logarithmic scale for background
        vmin, vmax: Minimum and maximum values for background colormap
        colorbar: Whether to add a colorbar for background
        title: Custom title
    
    Returns:
        fig, ax: The figure and axis objects
    """
    # Extract grid coordinates (now from tuple)
    x, y = mhd_system.grid
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Get vector field components based on type
    if field_type == 'velocity':
        u = mhd_system.velocity[0]
        v = mhd_system.velocity[1]
        field_name = 'Velocity Streamlines'
    elif field_type == 'magnetic':
        u = mhd_system.magnetic_field[0]
        v = mhd_system.magnetic_field[1]
        field_name = 'Magnetic Field Lines'
    else:
        raise ValueError(f"Unknown field type: {field_type}")
    
    # Create new figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Plot background field if specified
    if background is not None:
        bg_fig, _ = visualize_2d_scalar_field(
            mhd_system, background, time=None, ax=ax, 
            cmap=bg_cmap, log_scale=log_scale, vmin=vmin, 
            vmax=vmax, colorbar=colorbar, title=None
        )
    
    # Plot streamlines
    ax.streamplot(X.T, Y.T, u.T, v.T, density=density, color=color, linewidth=linewidth)
    
    # Set plot title
    if title is None:
        title = field_name
    if time is not None:
        title += f" at t = {time:.3f}"
    ax.set_title(title)
    
    # Set axis labels
    ax.set_xlabel(mhd_system.coord_names[0])
    ax.set_ylabel(mhd_system.coord_names[1])
    
    return fig, ax

def plot_mhd_diagnostics(mhd_system, time=None):
    """
    Create a comprehensive diagnostic plot for the MHD system state.
    
    Args:
        mhd_system: MHD system to visualize
        time: Simulation time
    
    Returns:
        fig: The figure object
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Set up grid of plots
    gs = fig.add_gridspec(3, 3)
    
    # Density plot
    ax1 = fig.add_subplot(gs[0, 0])
    visualize_2d_scalar_field(mhd_system, 'density', time=time, ax=ax1, colorbar=True)
    
    # Pressure plot
    ax2 = fig.add_subplot(gs[0, 1])
    visualize_2d_scalar_field(mhd_system, 'pressure', time=time, ax=ax2, colorbar=True)
    
    # Magnetic pressure (B^2/2) plot
    ax3 = fig.add_subplot(gs[0, 2])
    b_squared = mhd_system.magnetic_field[0]**2 + mhd_system.magnetic_field[1]**2
    X, Y = np.meshgrid(mhd_system.grid[0], mhd_system.grid[1], indexing='ij')
    im3 = ax3.pcolormesh(X, Y, b_squared/2, cmap='viridis')
    plt.colorbar(im3, ax=ax3)
    title = "Magnetic Pressure"
    if time is not None:
        title += f" at t = {time:.3f}"
    ax3.set_title(title)
    ax3.set_xlabel(mhd_system.coord_names[0])
    ax3.set_ylabel(mhd_system.coord_names[1])
    
    # Velocity magnitude with streamlines
    ax4 = fig.add_subplot(gs[1, 0])
    visualize_2d_scalar_field(mhd_system, 'velocity_magnitude', time=None, ax=ax4, colorbar=True)
    ax4.streamplot(X.T, Y.T, mhd_system.velocity[0].T, mhd_system.velocity[1].T, 
                  color='k', density=1.0, linewidth=0.7)
    title = "Velocity Field"
    if time is not None:
        title += f" at t = {time:.3f}"
    ax4.set_title(title)
    
    # Magnetic field magnitude with field lines
    ax5 = fig.add_subplot(gs[1, 1])
    visualize_2d_scalar_field(mhd_system, 'magnetic_magnitude', time=None, ax=ax5, colorbar=True)
    ax5.streamplot(X.T, Y.T, mhd_system.magnetic_field[0].T, mhd_system.magnetic_field[1].T, 
                  color='k', density=1.0, linewidth=0.7)
    title = "Magnetic Field"
    if time is not None:
        title += f" at t = {time:.3f}"
    ax5.set_title(title)
    
    # Current density plot
    ax6 = fig.add_subplot(gs[1, 2])
    visualize_2d_scalar_field(mhd_system, 'current_density', time=time, ax=ax6, 
                            cmap='seismic', vmin=None, vmax=None, colorbar=True)
    
    # Vorticity plot
    ax7 = fig.add_subplot(gs[2, 0])
    visualize_2d_scalar_field(mhd_system, 'vorticity', time=time, ax=ax7, 
                            cmap='seismic', vmin=None, vmax=None, colorbar=True)
    
    # Total pressure (thermal + magnetic)
    ax8 = fig.add_subplot(gs[2, 1])
    total_pressure = mhd_system.pressure + b_squared/2
    im8 = ax8.pcolormesh(X, Y, total_pressure, cmap='viridis')
    plt.colorbar(im8, ax=ax8)
    title = "Total Pressure"
    if time is not None:
        title += f" at t = {time:.3f}"
    ax8.set_title(title)
    ax8.set_xlabel(mhd_system.coord_names[0])
    ax8.set_ylabel(mhd_system.coord_names[1])
    
    # Plasma beta (ratio of thermal to magnetic pressure)
    ax9 = fig.add_subplot(gs[2, 2])
    beta = mhd_system.pressure / (b_squared/2 + 1e-10)  # avoid division by zero
    im9 = ax9.pcolormesh(X, Y, beta, cmap='viridis', norm=LogNorm(vmin=0.1, vmax=10))
    plt.colorbar(im9, ax=ax9)
    title = "Plasma Beta"
    if time is not None:
        title += f" at t = {time:.3f}"
    ax9.set_title(title)
    ax9.set_xlabel(mhd_system.coord_names[0])
    ax9.set_ylabel(mhd_system.coord_names[1])
    
    plt.tight_layout()
    return fig 