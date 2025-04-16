"""
Constrained Transport module for MHD simulations.

This module implements methods to maintain the divergence-free constraint
of the magnetic field (∇·B = 0) in MHD simulations using the constrained 
transport approach on a staggered grid.
"""

import numpy as np
from numba import njit

@njit
def initialize_face_centered_b(cell_centered_b, grid_shape):
    """
    Initialize face-centered magnetic field from cell-centered values.
    
    In constrained transport, magnetic field components are stored at cell faces,
    which naturally ensures div(B) = 0 to machine precision when evolved correctly.
    
    Args:
        cell_centered_b: List of cell-centered magnetic field components [Bx, By, Bz]
        grid_shape: Shape of the computational grid
        
    Returns:
        Face-centered magnetic field components
    """
    dimension = len(grid_shape)
    
    # Create arrays for face-centered fields
    # Each component is defined on the face perpendicular to its direction
    face_b = [None] * dimension
    
    # Convert cell-centered B to face-centered B with simple averaging
    for i in range(dimension):
        # Face field shapes have one extra point in the corresponding direction
        face_shape = list(grid_shape)
        face_shape[i] += 1
        face_b[i] = np.zeros(face_shape)
        
        # Interior points: average from cell centers
        slices_inner = [slice(1, s) for s in face_shape]
        slices_upper = [slice(None, s) for s in grid_shape]
        slices_lower = [slice(1, s+1) for s in grid_shape]
        
        # Special handling for the face component direction
        slices_inner[i] = slice(1, face_shape[i]-1)
        slices_upper[i] = slice(None, grid_shape[i])
        slices_lower[i] = slice(1, grid_shape[i]+1)
        
        # Averaging: B_face = (B_cell + B_cell+1) / 2
        face_b[i][tuple(slices_inner)] = (
            cell_centered_b[i][tuple(slices_upper)] + 
            cell_centered_b[i][tuple(slices_lower)]
        ) / 2.0
        
        # Boundary faces: just copy the nearest cell value
        # This is a simple approach - more sophisticated treatment possible
        for boundary in [0, face_shape[i]-1]:
            slices_boundary = [slice(None, s) for s in face_shape]
            slices_boundary[i] = boundary
            
            slices_nearest = [slice(None, s) for s in grid_shape]
            slices_nearest[i] = 0 if boundary == 0 else -1
            
            face_b[i][tuple(slices_boundary)] = cell_centered_b[i][tuple(slices_nearest)]
    
    return face_b

@njit
def compute_emf(velocity, magnetic_field, grid_spacing):
    """
    Compute the electromotive force (EMF) for constrained transport.
    
    The EMF is defined as E = v × B, and is used in Faraday's law to update
    the magnetic field: ∂B/∂t = -∇ × E
    
    In 3D, EMF has three components (Ex, Ey, Ez), each defined at cell edges.
    
    Args:
        velocity: List of velocity components [vx, vy, vz]
        magnetic_field: List of face-centered magnetic field components [Bx, By, Bz]
        grid_spacing: Grid spacing in each direction
        
    Returns:
        Electromotive force components at cell edges
    """
    # This implementation is for 3D
    # For 2D, the 3rd component patterns are simply adjusted
    
    # Extract grid shape from the first magnetic field component
    grid_shape = magnetic_field[0].shape
    dimension = len(grid_shape)
    
    # Create arrays for EMF components
    # In 3D, each EMF component is along a cell edge
    emf = [None] * dimension
    
    # Compute EMF at appropriate edge locations
    # Ex is along x-edges: Ex = vy*Bz - vz*By
    # Ey is along y-edges: Ey = vz*Bx - vx*Bz
    # Ez is along z-edges: Ez = vx*By - vy*Bx
    
    # For 3D:
    if dimension == 3:
        # Ex shape: (nx+1, ny, nz)
        emf_shape = list(grid_shape)
        emf_shape[1] -= 1
        emf_shape[2] -= 1
        emf[0] = np.zeros(emf_shape)
        
        # Ey shape: (nx, ny+1, nz)
        emf_shape = list(grid_shape)
        emf_shape[0] -= 1
        emf_shape[2] -= 1
        emf[1] = np.zeros(emf_shape)
        
        # Ez shape: (nx, ny, nz+1)
        emf_shape = list(grid_shape)
        emf_shape[0] -= 1
        emf_shape[1] -= 1
        emf[2] = np.zeros(emf_shape)
        
        # Compute EMF components by averaging velocity and B to edge locations
        # and then computing cross products
        # This is a placeholder - the actual implementation would need to 
        # carefully average the staggered values to the edge locations
        pass
    
    # For 2D (x-y plane):
    elif dimension == 2:
        # Only Ez component needed for 2D in x-y plane
        emf_shape = [grid_shape[0]-1, grid_shape[1]-1]
        emf[0] = np.zeros(emf_shape)  # Placeholder for Ex
        emf[1] = np.zeros(emf_shape)  # Placeholder for Ey
        
        # Ez at cell corners
        emf[2] = np.zeros(emf_shape)
        
        # Compute Ez = vx*By - vy*Bx at cell corners
        # Average velocity and B components to corner locations
        for i in range(emf_shape[0]):
            for j in range(emf_shape[1]):
                # Average velocities to the corner
                vx_corner = 0.25 * (
                    velocity[0][i, j] + velocity[0][i, j+1] + 
                    velocity[0][i+1, j] + velocity[0][i+1, j+1]
                )
                vy_corner = 0.25 * (
                    velocity[1][i, j] + velocity[1][i, j+1] + 
                    velocity[1][i+1, j] + velocity[1][i+1, j+1]
                )
                
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
    # Extract grid shapes
    dimension = len(face_b)
    
    # Create arrays for updated fields
    updated_face_b = [np.copy(face_b[i]) for i in range(dimension)]
    
    # Update face-centered B fields using curl of EMF
    # For 3D:
    if dimension == 3:
        # Update Bx: ∂Bx/∂t = ∂Ez/∂y - ∂Ey/∂z
        # Update By: ∂By/∂t = ∂Ex/∂z - ∂Ez/∂x
        # Update Bz: ∂Bz/∂t = ∂Ey/∂x - ∂Ex/∂y
        # This is a placeholder - the actual implementation would need to
        # carefully compute these derivatives on the staggered grid
        pass
    
    # For 2D (x-y plane):
    elif dimension == 2:
        dx, dy = grid_spacing
        
        # Update Bx using Ez: ∂Bx/∂t = -∂Ez/∂y
        for i in range(face_b[0].shape[0]):
            for j in range(1, face_b[0].shape[1]):
                # Finite difference for ∂Ez/∂y
                dez_dy = (emf[2][i-1, j-1] - emf[2][i-1, j-2]) / dy
                updated_face_b[0][i, j] = face_b[0][i, j] - dt * dez_dy
        
        # Update By using Ez: ∂By/∂t = ∂Ez/∂x
        for i in range(1, face_b[1].shape[0]):
            for j in range(face_b[1].shape[1]):
                # Finite difference for ∂Ez/∂x
                dez_dx = (emf[2][i-1, j-1] - emf[2][i-2, j-1]) / dx
                updated_face_b[1][i, j] = face_b[1][i, j] + dt * dez_dx
    
    return updated_face_b

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
    
    # Determine grid shape from face-centered fields
    cell_shape = [face_b[i].shape[i] - 1 for i in range(dimension)]
    
    # Create arrays for cell-centered fields
    cell_b = [np.zeros(cell_shape) for _ in range(dimension)]
    
    # Average face values to cell centers
    for i in range(dimension):
        for idx in np.ndindex(*cell_shape):
            # Create index arrays for the two adjacent faces
            face_idx_lower = list(idx)
            face_idx_upper = list(idx)
            face_idx_upper[i] += 1
            
            # Average the two face values
            cell_b[i][idx] = 0.5 * (
                face_b[i][tuple(face_idx_lower)] + 
                face_b[i][tuple(face_idx_upper)]
            )
    
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
    div_shape = [face_b[i].shape[i] - 1 for i in range(dimension)]
    
    # Create array for divergence
    div_b = np.zeros(div_shape)
    
    # Compute divergence at cell centers using face values
    # For 3D: div(B) = (Bx(i+1/2) - Bx(i-1/2))/dx + (By(j+1/2) - By(j-1/2))/dy + (Bz(k+1/2) - Bz(k-1/2))/dz
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
    dimension = len(grid)
    
    # For 2D (x-y plane), we need only A_z component
    if dimension == 2:
        x, y = grid
        dx, dy = grid_spacing
        
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
        
        # Return face-centered B as a list
        face_b = [Bx, By]
        
    # For 3D, implement the full curl of A
    else:
        # Implementation for 3D would compute the curl of A = [A_x, A_y, A_z]
        # B_x = ∂A_z/∂y - ∂A_y/∂z
        # B_y = ∂A_x/∂z - ∂A_z/∂x
        # B_z = ∂A_y/∂x - ∂A_x/∂y
        
        # This is a placeholder - would need complete implementation for 3D
        face_b = [np.zeros((1, 1, 1)) for _ in range(dimension)]
        
    return face_b 