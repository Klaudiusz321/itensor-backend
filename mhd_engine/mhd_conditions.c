/**
 * mhd_conditions.c - Boundary conditions for MHD simulation
 * 
 * Implements various boundary conditions for MHD simulations including
 * open, closed, reflective, and periodic boundaries, with support for
 * a user‐supplied custom boundary function.
 */

#include "mhd.h"

/**
 * Set the boundary conditions type for the simulation
 */
void mhd_set_boundary_type(MHDSimulation *sim, BoundaryType type) {
    if (!sim) return;
    sim->boundary_type = type;
}

/**
 * Register a custom boundary function and set boundary type to CUSTOM
 */
void mhd_set_custom_boundaries(MHDSimulation *sim,
                               CustomBoundaryFunc func)
{
    if (!sim) return;
    sim->custom_boundary_func = func;
    sim->boundary_type        = CUSTOM;
}

/**
 * Apply boundary conditions based on the current simulation settings
 */
void mhd_apply_boundary_conditions(MHDSimulation *sim, GridCell ***buf) {
    if (!sim || !buf) return;

    switch (sim->boundary_type) {
        case OPEN:
            mhd_set_open_boundaries(sim, buf);
            break;
        case CLOSED:
            mhd_set_closed_boundaries(sim, buf);
            break;
        case REFLECTIVE:
            mhd_set_reflective_boundaries(sim, buf);
            break;
        case PERIODIC:
            mhd_set_periodic_boundaries(sim, buf);
            break;
        case CUSTOM:
            if (sim->custom_boundary_func) {
                /* assume custom func now has signature
                   void (*custom_boundary_func)(MHDSimulation *, GridCell ***);
                */
                sim->custom_boundary_func(sim, buf);
            }
            break;
        default:
            mhd_set_periodic_boundaries(sim, buf);
            break;
    }
}

/**
 * Set open (zero-gradient) boundary conditions
 */
void mhd_set_open_boundaries(MHDSimulation *sim) {
    if (!sim) return;
    
    int nx = sim->grid_size_x, ny = sim->grid_size_y, nz = sim->grid_size_z;
    // X boundaries
    for (int j = 0; j < ny; j++)
    for (int k = 0; k < nz; k++) {
        sim->grid[0][j][k]    = sim->grid[1][j][k];
        sim->grid[nx-1][j][k] = sim->grid[nx-2][j][k];
    }
    // Y boundaries
    for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) {
        sim->grid[i][0][k]    = sim->grid[i][1][k];
        sim->grid[i][ny-1][k] = sim->grid[i][ny-2][k];
    }
    // Z boundaries
    if (nz > 1) {
        for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            sim->grid[i][j][0]    = sim->grid[i][j][1];
            sim->grid[i][j][nz-1] = sim->grid[i][j][nz-2];
        }
    }
}

/**
 * Set closed (impenetrable) boundary conditions: zero normal velocity,
 * zero-gradient for all other quantities (including B)
 */
void mhd_set_closed_boundaries(MHDSimulation *sim) {
    if (!sim) return;
    
    int nx = sim->grid_size_x, ny = sim->grid_size_y, nz = sim->grid_size_z;
    // X boundaries
    for (int j = 0; j < ny; j++)
    for (int k = 0; k < nz; k++) {
        // copy all fields, then zero normal velocity
        sim->grid[0][j][k]    = sim->grid[1][j][k];
        sim->grid[0][j][k].velocity.x = 0.0;
        sim->grid[nx-1][j][k] = sim->grid[nx-2][j][k];
        sim->grid[nx-1][j][k].velocity.x = 0.0;
    }
    // Y boundaries
    for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) {
        sim->grid[i][0][k]    = sim->grid[i][1][k];
        sim->grid[i][0][k].velocity.y = 0.0;
        sim->grid[i][ny-1][k] = sim->grid[i][ny-2][k];
        sim->grid[i][ny-1][k].velocity.y = 0.0;
    }
    // Z boundaries
    if (nz > 1) {
        for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            sim->grid[i][j][0]    = sim->grid[i][j][1];
            sim->grid[i][j][0].velocity.z = 0.0;
            sim->grid[i][j][nz-1] = sim->grid[i][j][nz-2];
            sim->grid[i][j][nz-1].velocity.z = 0.0;
        }
    }
}

/**
 * Set reflective boundary conditions: reverse normal velocity,
 * invert tangential magnetic field, keep normal B continuous.
 */
void mhd_set_reflective_boundaries(MHDSimulation *sim) {
    if (!sim) return;
    
    int nx = sim->grid_size_x, ny = sim->grid_size_y, nz = sim->grid_size_z;
    // X boundaries (normal = x, tangential = y,z)
    for (int j = 0; j < ny; j++)
    for (int k = 0; k < nz; k++) {
        // left face
        sim->grid[0][j][k]    = sim->grid[1][j][k];
        sim->grid[0][j][k].velocity.x  = -sim->grid[1][j][k].velocity.x;
        sim->grid[0][j][k].magnetic.y  = -sim->grid[1][j][k].magnetic.y;
        sim->grid[0][j][k].magnetic.z  = -sim->grid[1][j][k].magnetic.z;
        // right face
        sim->grid[nx-1][j][k] = sim->grid[nx-2][j][k];
        sim->grid[nx-1][j][k].velocity.x = -sim->grid[nx-2][j][k].velocity.x;
        sim->grid[nx-1][j][k].magnetic.y = -sim->grid[nx-2][j][k].magnetic.y;
        sim->grid[nx-1][j][k].magnetic.z = -sim->grid[nx-2][j][k].magnetic.z;
    }
    // Y boundaries (normal = y, tangential = x,z)
    for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) {
        sim->grid[i][0][k]    = sim->grid[i][1][k];
        sim->grid[i][0][k].velocity.y  = -sim->grid[i][1][k].velocity.y;
        sim->grid[i][0][k].magnetic.x  = -sim->grid[i][1][k].magnetic.x;
        sim->grid[i][0][k].magnetic.z  = -sim->grid[i][1][k].magnetic.z;
        sim->grid[i][ny-1][k] = sim->grid[i][ny-2][k];
        sim->grid[i][ny-1][k].velocity.y = -sim->grid[i][ny-2][k].velocity.y;
        sim->grid[i][ny-1][k].magnetic.x = -sim->grid[i][ny-2][k].magnetic.x;
        sim->grid[i][ny-1][k].magnetic.z = -sim->grid[i][ny-2][k].magnetic.z;
    }
    // Z boundaries (normal = z, tangential = x,y)
    if (nz > 1) {
        for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            sim->grid[i][j][0]    = sim->grid[i][j][1];
            sim->grid[i][j][0].velocity.z  = -sim->grid[i][j][1].velocity.z;
            sim->grid[i][j][0].magnetic.x  = -sim->grid[i][j][1].magnetic.x;
            sim->grid[i][j][0].magnetic.y  = -sim->grid[i][j][1].magnetic.y;
            sim->grid[i][j][nz-1] = sim->grid[i][j][nz-2];
            sim->grid[i][j][nz-1].velocity.z = -sim->grid[i][j][nz-2].velocity.z;
            sim->grid[i][j][nz-1].magnetic.x = -sim->grid[i][j][nz-2].magnetic.x;
            sim->grid[i][j][nz-1].magnetic.y = -sim->grid[i][j][nz-2].magnetic.y;
        }
    }
}

/**
 * Set periodic boundary conditions
 */
void mhd_set_periodic_boundaries(MHDSimulation *sim) {
    if (!sim) return;
    
    int nx = sim->grid_size_x, ny = sim->grid_size_y, nz = sim->grid_size_z;
    // X boundaries
    for (int j = 0; j < ny; j++)
    for (int k = 0; k < nz; k++) {
        sim->grid[0][j][k]    = sim->grid[nx-2][j][k];
        sim->grid[nx-1][j][k] = sim->grid[1][j][k];
    }
    // Y boundaries
    for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) {
        sim->grid[i][0][k]    = sim->grid[i][ny-2][k];
        sim->grid[i][ny-1][k] = sim->grid[i][1][k];
    }
    // Z boundaries
    if (nz > 1) {
        for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            sim->grid[i][j][0]    = sim->grid[i][j][nz-2];
            sim->grid[i][j][nz-1] = sim->grid[i][j][1];
        }
    }
}
