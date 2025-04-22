"""
Constrained Transport module for MHD simulations.

This module implements methods to maintain the divergence-free constraint
of the magnetic field (∇·B = 0) in MHD simulations using the constrained 
transport approach on a staggered grid.
"""

import numpy as np
from numba import njit
from numba.typed import List
from numba.typed import List as TypedList

@njit
def initialize_face_centered_b(cell_centered_b, grid_shape):
    """
    Initialize face-centered magnetic field from cell-centered values.
    
    This creates a staggered grid for magnetic field components to maintain
    the div(B) = 0 constraint. In 2D, Bx is at vertical faces, By is at horizontal faces.
    
    Args:
        cell_centered_b: List of cell-centered magnetic field components
        grid_shape: Tuple of grid dimensions (nx, ny) or (nx, ny, nz)
    
    Returns:
        List of face-centered magnetic field components
    """
    dimension = len(grid_shape)
    
    # Create a typed List for face-centered B components
    face_b = List()
    
    # Handle 2D case explicitly with static indexing
    if dimension == 2:
        nx, ny = grid_shape
        
        # Allocate face-centered arrays
        # Bx: Add one cell in x-direction
        bx_face = np.zeros((nx + 1, ny))
        # By: Add one cell in y-direction
        by_face = np.zeros((nx, ny + 1))
        
        # Add arrays to the list
        face_b.append(bx_face)
        face_b.append(by_face)
        
        # Fill interior points by averaging cell centers
        # For Bx faces (normal to x-direction) - internal x-faces
        for i in range(1, nx):
            for j in range(ny):
                face_b[0][i, j] = 0.5 * (cell_centered_b[0][i-1, j] + cell_centered_b[0][i, j])
        
        # For By faces (normal to y-direction) - internal y-faces
        for i in range(nx):
            for j in range(1, ny):
                face_b[1][i, j] = 0.5 * (cell_centered_b[1][i, j-1] + cell_centered_b[1][i, j])
        
        # Handle boundaries - Bx faces
        # Left boundary (x=0)
        for j in range(ny):
            face_b[0][0, j] = cell_centered_b[0][0, j]
        # Right boundary (x=nx)
        for j in range(ny):
            face_b[0][nx, j] = cell_centered_b[0][nx-1, j]
            
        # Handle boundaries - By faces
        # Bottom boundary (y=0)
        for i in range(nx):
            face_b[1][i, 0] = cell_centered_b[1][i, 0]
        # Top boundary (y=ny)
        for i in range(nx):
            face_b[1][i, ny] = cell_centered_b[1][i, ny-1]
            
    elif dimension == 3:
        nx, ny, nz = grid_shape
        
        # Allocate face-centered arrays
        bx_face = np.zeros((nx + 1, ny, nz))
        by_face = np.zeros((nx, ny + 1, nz))
        bz_face = np.zeros((nx, ny, nz + 1))
        
        # Add arrays to the list
        face_b.append(bx_face)
        face_b.append(by_face)
        face_b.append(bz_face)
        
        # Fill interior points by averaging cell centers
        # For Bx faces (normal to x-direction)
        for i in range(1, nx):
            for j in range(ny):
                for k in range(nz):
                    face_b[0][i, j, k] = 0.5 * (cell_centered_b[0][i-1, j, k] + cell_centered_b[0][i, j, k])
        
        # For By faces (normal to y-direction)
        for i in range(nx):
            for j in range(1, ny):
                for k in range(nz):
                    face_b[1][i, j, k] = 0.5 * (cell_centered_b[1][i, j-1, k] + cell_centered_b[1][i, j, k])
        
        # For Bz faces (normal to z-direction)
        for i in range(nx):
            for j in range(ny):
                for k in range(1, nz):
                    face_b[2][i, j, k] = 0.5 * (cell_centered_b[2][i, j, k-1] + cell_centered_b[2][i, j, k])
        
        # Handle boundaries - Bx faces
        # Left boundary (x=0)
        for j in range(ny):
            for k in range(nz):
                face_b[0][0, j, k] = cell_centered_b[0][0, j, k]
        # Right boundary (x=nx)
        for j in range(ny):
            for k in range(nz):
                face_b[0][nx, j, k] = cell_centered_b[0][nx-1, j, k]
        
        # Handle boundaries - By faces
        # Bottom boundary (y=0)
        for i in range(nx):
            for k in range(nz):
                face_b[1][i, 0, k] = cell_centered_b[1][i, 0, k]
        # Top boundary (y=ny)
        for i in range(nx):
            for k in range(nz):
                face_b[1][i, ny, k] = cell_centered_b[1][i, ny-1, k]
        
        # Handle boundaries - Bz faces
        # Front boundary (z=0)
        for i in range(nx):
            for j in range(ny):
                face_b[2][i, j, 0] = cell_centered_b[2][i, j, 0]
        # Back boundary (z=nz)
        for i in range(nx):
            for j in range(ny):
                face_b[2][i, j, nz] = cell_centered_b[2][i, j, nz-1]
    
    return face_b

def compute_emf(velocity, magnetic_field, grid_spacing=None):
    """
    Compute the electromotive force (EMF) for constrained transport.
    
    The EMF is defined as E = v × B, and is used in Faraday's law to update
    the magnetic field: ∂B/∂t = -∇ × E
    
    In 3D, EMF has three components (Ex, Ey, Ez), each defined at cell edges.
    
    Args:
        velocity: List of velocity components [vx, vy, vz]
        magnetic_field: List of face-centered magnetic field components [Bx, By, Bz]
        grid_spacing: Grid spacing in each direction (optional, not used in current implementation)
        
    Returns:
        Electromotive force components at cell edges
    """
    # Extract grid shape from the first magnetic field component
    dimension = len(magnetic_field)
    
    # Create typed List for EMF components
    emf = List()
    
    # Compute EMF at appropriate edge locations
    # Ex is along x-edges: Ex = vy*Bz - vz*By
    # Ey is along y-edges: Ey = vz*Bx - vx*Bz
    # Ez is along z-edges: Ez = vx*By - vy*Bx
    
    # For 3D:
    if dimension == 3:
        # Get grid dimensions from the magnetic field components
        nx = magnetic_field[0].shape[0] - 1
        ny = magnetic_field[1].shape[1] - 1
        nz = magnetic_field[2].shape[2] - 1
        
        # Ex shape: (nx+1, ny, nz)
        emf.append(np.zeros((nx+1, ny-1, nz-1)))
        
        # Ey shape: (nx, ny+1, nz)
        emf.append(np.zeros((nx-1, ny+1, nz-1)))
        
        # Ez shape: (nx, ny, nz+1)
        emf.append(np.zeros((nx-1, ny-1, nz+1)))
        
        # Compute EMF components by averaging velocity and B to edge locations
        # and then computing cross products
        # This is a placeholder - the actual implementation would need to 
        # carefully average the staggered values to the edge locations
        pass
    
    # For 2D (x-y plane):
    elif dimension == 2:
        # Get grid dimensions from the magnetic field components and velocity
        nx = min(magnetic_field[0].shape[0] - 1, velocity[0].shape[0] - 1)  # Safe size for emf
        ny = min(magnetic_field[1].shape[1] - 1, velocity[0].shape[1] - 1)
        
        # For 2D, we only need Ez (out-of-plane) component
        emf_shape_x = np.zeros((1, 1))  # Placeholder for Ex (not used in 2D)
        emf_shape_y = np.zeros((1, 1))  # Placeholder for Ey (not used in 2D)
        
        # Ez at cell corners: (nx, ny)
        emf_shape_z = np.zeros((nx, ny))
        
        # Add placeholders for completeness
        emf.append(emf_shape_x)
        emf.append(emf_shape_y)
        emf.append(emf_shape_z)
        
        # Compute Ez = vx*By - vy*Bx at cell corners
        # Average velocity and B components to corner locations
        for i in range(nx):
            for j in range(ny):
                # Make sure we don't go out of bounds when i+1 or j+1 is accessed
                if i+1 >= velocity[0].shape[0] or j+1 >= velocity[0].shape[1]:
                    continue
                    
                # Average velocities to the corner
                vx_corner = 0.25 * (
                    velocity[0][i, j] + velocity[0][i, j+1] + 
                    velocity[0][i+1, j] + velocity[0][i+1, j+1]
                )
                vy_corner = 0.25 * (
                    velocity[1][i, j] + velocity[1][i, j+1] + 
                    velocity[1][i+1, j] + velocity[1][i+1, j+1]
                )
                
                # Make sure we don't go out of bounds for magnetic field components
                if j+1 >= magnetic_field[0].shape[1] or i+1 >= magnetic_field[1].shape[0]:
                    continue
                    
                # Get magnetic field components at the corner
                # For 2D, Bx is at vertical faces, By is at horizontal faces
                bx_corner = 0.5 * (magnetic_field[0][i, j] + magnetic_field[0][i, j+1])
                by_corner = 0.5 * (magnetic_field[1][i, j] + magnetic_field[1][i+1, j])
                
                # Compute Ez = vx*By - vy*Bx
                emf[2][i, j] = vx_corner * by_corner - vy_corner * bx_corner
    
    return emf

@njit
def update_face_centered_b(face_b, emf, dt, grid_spacing):
    """
    Update the face-centered magnetic field using constrained transport.
    
    The update follows Faraday's law: ∂B/∂t = -∇ × E
    Where E is the electromotive force (EMF).
    
    Args:
        face_b: List of face-centered magnetic field components
        emf: Electromotive force components at cell edges
        dt: Time step
        grid_spacing: Grid spacing in each direction
        
    Returns:
        Updated face-centered magnetic field components
    """
    # Extract the dimension
    dimension = len(face_b)
    
    # Create typed List for updated fields
    updated_face_b = List()
    for i in range(dimension):
        updated_face_b.append(np.copy(face_b[i]))
    
    # Update face-centered B fields using curl of EMF
    if dimension == 3:
        # 3D update logic requires careful indexing for each component
        # This is a placeholder for the full 3D implementation
        # Update Bx: ∂Bx/∂t = ∂Ez/∂y - ∂Ey/∂z
        # Update By: ∂By/∂t = ∂Ex/∂z - ∂Ez/∂x
        # Update Bz: ∂Bz/∂t = ∂Ey/∂x - ∂Ex/∂y
        pass
    
    # For 2D (x-y plane):
    elif dimension == 2:
        if len(grid_spacing) >= 2:
            dx, dy = grid_spacing[0], grid_spacing[1]
        else:
            # Fall back if grid_spacing doesn't have enough elements
            dx = dy = grid_spacing[0]
        
        # Get the dimensions
        nx = face_b[0].shape[0]
        ny = face_b[1].shape[1]
        
        # Update Bx using Ez: ∂Bx/∂t = -∂Ez/∂y
        for i in range(nx):
            for j in range(1, ny):
                # Check bounds before accessing emf array
                if i >= emf[2].shape[0] or j-1 >= emf[2].shape[1] or j-2 >= emf[2].shape[1]:
                    continue
                # Finite difference for ∂Ez/∂y
                dez_dy = (emf[2][i, j-1] - emf[2][i, j-2]) / dy
                updated_face_b[0][i, j] = face_b[0][i, j] - dt * dez_dy
        
        # Update By using Ez: ∂By/∂t = ∂Ez/∂x
        for i in range(1, nx):
            for j in range(ny):
                # Check bounds before accessing emf array
                if i-1 >= emf[2].shape[0] or i-2 >= emf[2].shape[0] or j >= emf[2].shape[1]:
                    continue
                # Finite difference for ∂Ez/∂x
                dez_dx = (emf[2][i-1, j] - emf[2][i-2, j]) / dx
                updated_face_b[1][i, j] = face_b[1][i, j] + dt * dez_dx
    
    return updated_face_b

@njit
def face_to_cell_centered_b(face_b):
    """
    Convert face-centered magnetic field to cell-centered values.
    
    This is used for visualization or when cell-centered values are needed for
    calculations such as Lorentz force or pressure.
    
    Args:
        face_b: List of face-centered magnetic field components
        
    Returns:
        Cell-centered magnetic field components
    """
    dimension = len(face_b)
    
    # For now, only support 2D case with Numba
    if dimension == 2:
        return face_to_cell_centered_b_2d(face_b)
    else:
        # Since we're focusing on 2D for now, bypass 3D case
        # This should be replaced with proper 3D implementation later
        cell_b = List()
        for i in range(dimension):
            cell_b.append(face_b[i].copy())
        return cell_b

@njit
def face_to_cell_centered_b_2d(face_b):
    """Helper function for 2D case to avoid dimension-specific code in main function."""
    # Determine grid shape from face-centered fields
    nx = face_b[0].shape[0] - 1  # x-direction length
    ny = face_b[1].shape[1] - 1  # y-direction length
    
    # Create a typed List for cell-centered fields
    cell_b = List()
    
    # Initialize output arrays
    cell_b.append(np.zeros((nx, ny)))
    cell_b.append(np.zeros((nx, ny)))
    
    # Average face values to cell centers - manual indexing for 2D
    # For Bx (in x-direction)
    for i in range(nx):
        for j in range(ny):
            cell_b[0][i, j] = 0.5 * (face_b[0][i, j] + face_b[0][i+1, j])
    
    # For By (in y-direction)
    for i in range(nx):
        for j in range(ny):
            cell_b[1][i, j] = 0.5 * (face_b[1][i, j] + face_b[1][i, j+1])
    
    return cell_b

@njit
def face_to_cell_centered_b_3d(face_b):
    """Helper function for 3D case to avoid dimension-specific code in main function."""
    # Determine grid shape from face-centered fields
    nx = face_b[0].shape[0] - 1
    ny = face_b[1].shape[1] - 1
    nz = face_b[2].shape[2] - 1
    
    # Create a typed List for cell-centered fields
    cell_b = List()
    
    # Initialize output arrays
    cell_b.append(np.zeros((nx, ny, nz)))
    cell_b.append(np.zeros((nx, ny, nz)))
    cell_b.append(np.zeros((nx, ny, nz)))
    
    # Average face values to cell centers - manual indexing for 3D
    # For Bx (in x-direction)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell_b[0][i, j, k] = 0.5 * (face_b[0][i, j, k] + face_b[0][i+1, j, k])
    
    # For By (in y-direction)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell_b[1][i, j, k] = 0.5 * (face_b[1][i, j, k] + face_b[1][i, j+1, k])
    
    # For Bz (in z-direction)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell_b[2][i, j, k] = 0.5 * (face_b[2][i, j, k] + face_b[2][i, j, k+1])
    
    return cell_b

def check_divergence_free(face_b, grid_spacing):
    """
    Check if the magnetic field is divergence-free by computing div(B) on the staggered grid.
    
    Args:
        face_b: List of face-centered magnetic field components
        grid_spacing: Grid spacing in each direction
        
    Returns:
        Maximum absolute value of divergence
    """
    dimension = len(face_b)
    
    # Determine grid shape for divergence (same as cell centers)
    # Build div_shape directly as a statically typed tuple from scalars
    if dimension == 2:
        nx = face_b[0].shape[0] - 1  # x-direction length
        ny = face_b[1].shape[1] - 1  # y-direction length
        div_shape = (nx, ny)
    else:  # 3D case
        # Create separate function for 3D to avoid Numba seeing face_b[2] during 2D compilation
        return check_divergence_free_3d(face_b, grid_spacing)
    
    # Create array for divergence
    div_b = np.zeros(div_shape)
    
    # Compute divergence at cell centers using face values
    # For 2D: div(B) = (Bx(i+1/2) - Bx(i-1/2))/dx + (By(j+1/2) - By(j-1/2))/dy
    
    for idx in np.ndindex(*div_shape):
        for i in range(dimension):
            # Create index arrays for the two adjacent faces
            face_idx_lower = list(idx)
            face_idx_upper = list(idx)
            face_idx_upper[i] += 1
            
            # Add contribution from this direction to divergence
            div_b[idx] += (
                face_b[i][tuple(face_idx_upper)] - 
                face_b[i][tuple(face_idx_lower)]
            ) / grid_spacing[i]
    
    return np.max(np.abs(div_b))

def check_divergence_free_3d(face_b, grid_spacing):
    """
    Check if the magnetic field is divergence-free in 3D.
    
    This is a separate function to avoid Numba compilation issues with 2D simulations.
    
    Args:
        face_b: List of face-centered magnetic field components (3D)
        grid_spacing: Grid spacing in each direction
        
    Returns:
        Maximum absolute value of divergence
    """
    # For 3D, we know we have 3 components
    nx = face_b[0].shape[0] - 1
    ny = face_b[1].shape[1] - 1
    nz = face_b[2].shape[2] - 1
    div_shape = (nx, ny, nz)
    
    # Create array for divergence
    div_b = np.zeros(div_shape)
    
    # Compute divergence at cell centers using face values
    # div(B) = (Bx(i+1/2) - Bx(i-1/2))/dx + (By(j+1/2) - By(j-1/2))/dy + (Bz(k+1/2) - Bz(k-1/2))/dz
    
    for idx in np.ndindex(*div_shape):
        for i in range(3):  # 3 dimensions for 3D
            # Create index arrays for the two adjacent faces
            face_idx_lower = list(idx)
            face_idx_upper = list(idx)
            face_idx_upper[i] += 1
            
            # Add contribution from this direction to divergence
            div_b[idx] += (
                face_b[i][tuple(face_idx_upper)] - 
                face_b[i][tuple(face_idx_lower)]
            ) / grid_spacing[i]
    
    return np.max(np.abs(div_b))

def initialize_from_vector_potential(vector_potential_func, grid, grid_spacing):
    """
    Initialize face-centered magnetic field from a vector potential function.
    
    The magnetic field is computed as B = ∇ × A, where A is the vector potential.
    This automatically ensures div(B) = 0 to machine precision.
    
    Args:
        vector_potential_func: Function or list of functions that return vector potential components.
                             For 2D: A single function for A_z component is sufficient.
                             For 3D: A list of 3 functions for [A_x, A_y, A_z] components.
        grid: Grid coordinates as returned by create_grid
        grid_spacing: Grid spacing in each direction
        
    Returns:
        Face-centered magnetic field components
    """
    # Check that grid is a sequence (tuple, list, etc.)
    if not hasattr(grid, '__len__'):
        raise TypeError("Grid must be a sequence (tuple, list) of coordinate arrays")
    
    dimension = len(grid)
    
    # For 2D (x-y plane), we need only A_z component
    if dimension == 2:
        # Ensure grid is properly formatted
        if isinstance(grid, (list, tuple)) and len(grid) == 2:
            x, y = grid
        else:
            raise ValueError("For 2D, grid must be a sequence of 2 coordinate arrays")
            
        # Ensure grid_spacing is properly formatted
        if isinstance(grid_spacing, (list, tuple, dict)):
            if isinstance(grid_spacing, dict):
                # Extract spacing values if it's a dictionary
                dx = grid_spacing.get('x', grid_spacing.get(0, None))
                dy = grid_spacing.get('y', grid_spacing.get(1, None))
                if dx is None or dy is None:
                    raise ValueError("grid_spacing dictionary must contain 'x'/'y' or 0/1 keys")
            else:
                # Use as list/tuple
                if len(grid_spacing) >= 2:
                    dx, dy = grid_spacing[0], grid_spacing[1]
                else:
                    raise ValueError("grid_spacing must have at least 2 elements for 2D")
        else:
            raise TypeError("grid_spacing must be a sequence or dictionary")
        
        # Define vertices (corners) coordinates for A_z
        # In 2D, A_z is defined at cell corners (vertices)
        x_vertices = np.copy(x)  # Cell corners in x
        y_vertices = np.copy(y)  # Cell corners in y
        
        # Compute A_z at vertices
        A_z = np.zeros((len(x_vertices), len(y_vertices)))
        for i in range(len(x_vertices)):
            for j in range(len(y_vertices)):
                A_z[i, j] = vector_potential_func(x_vertices[i], y_vertices[j])
        
        # Create face-centered B arrays with appropriate shapes
        # Bx is defined at vertical faces: (i, j+1/2)
        Bx = np.zeros((len(x_vertices), len(y_vertices)-1))
        
        # By is defined at horizontal faces: (i+1/2, j)
        By = np.zeros((len(x_vertices)-1, len(y_vertices)))
        
        # Compute Bx = ∂A_z/∂y at vertical faces
        for i in range(len(x_vertices)):
            for j in range(len(y_vertices)-1):
                # Centered finite difference for ∂A_z/∂y
                Bx[i, j] = (A_z[i, j+1] - A_z[i, j]) / dy
        
        # Compute By = -∂A_z/∂x at horizontal faces
        for i in range(len(x_vertices)-1):
            for j in range(len(y_vertices)):
                # Centered finite difference for -∂A_z/∂x
                By[i, j] = -(A_z[i+1, j] - A_z[i, j]) / dx
        
        # Return face-centered B as a typed List
        face_b = List()
        face_b.append(Bx)
        face_b.append(By)
        
    # For 3D, implement the full curl of A
    else:
        # Implementation for 3D would compute the curl of A = [A_x, A_y, A_z]
        # B_x = ∂A_z/∂y - ∂A_y/∂z
        # B_y = ∂A_x/∂z - ∂A_z/∂x
        # B_z = ∂A_y/∂x - ∂A_x/∂y
        
        # This is a placeholder - would need complete implementation for 3D
        face_b = List()
        for _ in range(dimension):
            face_b.append(np.zeros((1, 1, 1)))
        
    return face_b 