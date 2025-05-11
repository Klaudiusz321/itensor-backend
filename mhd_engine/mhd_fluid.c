/**
 * mhd_fluid.c - Fluid dynamics functions for MHD simulation
 * 
 * Implements functions for managing fluid parameters (density, pressure, 
 * temperature, velocity) in the MHD simulation.
 */

#include "mhd.h"



#include "mhd.h"

/**
 * Update only the fluid quantities (density, pressure, temperature, velocity)
 * for one time step using the derivatives computed by mhd_update_derivatives().
 */
void mhd_update_fluid_dynamics(MHDSimulation *sim) {
    if (!sim) return;

    // wykorzystamy temp_grid do przechowania d(rho)/dt, dP/dt, dT/dt, dV/dt
    mhd_update_derivatives(sim, sim->grid, sim->temp_grid);

    double h = sim->time_step;
    int nx = sim->grid_size_x,
        ny = sim->grid_size_y,
        nz = sim->grid_size_z;

    // Dla każdego wnętrza siatki (pomiń brzeg)
    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
            for (int k = 1; k < nz - 1; k++) {
                GridCell *c  = &sim->grid[i][j][k];
                GridCell *d  = &sim->temp_grid[i][j][k];

                // dρ/dt → ρ_new = ρ + h * dρ/dt
                c->density     += h * d->density;
                // dP/dt → P_new = P + h * dP/dt
                c->pressure    += h * d->pressure;
                // dT/dt → T_new = T + h * dT/dt
                c->temperature += h * d->temperature;

                // dV/dt → v_new = v + h * dv/dt
                c->velocity.x += h * d->velocity.x;
                c->velocity.y += h * d->velocity.y;
                c->velocity.z += h * d->velocity.z;

                // zabezpieczenie przed wartościami ujemnymi
                if (c->density <  0.001) c->density     = 0.001;
                if (c->pressure < 0.001) c->pressure    = 0.001;
                if (c->temperature < 0.001) c->temperature = 0.001;
            }
        }
    }
}



/**
 * Initialize fluid parameters with uniform values
 */
void mhd_initialize_fluid_parameters(MHDSimulation *sim, double density, double pressure, double temperature) {
    if (!sim) return;
    
    // Check for valid parameters
    if (density <= 0.0) {
        fprintf(stderr, "Warning: Density must be positive, using default value\n");
        density = 1.0;
    }
    
    if (pressure <= 0.0) {
        fprintf(stderr, "Warning: Pressure must be positive, using default value\n");
        pressure = 1.0;
    }
    
    if (temperature <= 0.0) {
        fprintf(stderr, "Warning: Temperature must be positive, using default value\n");
        temperature = 1.0;
    }
    
    // Set the initial values in the simulation struct
    sim->initial_density = density;
    sim->initial_pressure = pressure;
    sim->initial_temperature = temperature;
    
    // Set the values in all grid cells
    for (int i = 0; i < sim->grid_size_x; i++) {
        for (int j = 0; j < sim->grid_size_y; j++) {
            for (int k = 0; k < sim->grid_size_z; k++) {
                sim->grid[i][j][k].density = density;
                sim->grid[i][j][k].pressure = pressure;
                sim->grid[i][j][k].temperature = temperature;
            }
        }
    }
}

/**
 * Set the initial velocity field
 */
void mhd_set_initial_velocity(MHDSimulation *sim, double vx, double vy, double vz) {
    if (!sim) return;
    
    // Set the initial velocity in the simulation struct
    sim->initial_velocity.x = vx;
    sim->initial_velocity.y = vy;
    sim->initial_velocity.z = vz;
    
    // Set the velocity in all grid cells
    for (int i = 0; i < sim->grid_size_x; i++) {
        for (int j = 0; j < sim->grid_size_y; j++) {
            for (int k = 0; k < sim->grid_size_z; k++) {
                sim->grid[i][j][k].velocity.x = vx;
                sim->grid[i][j][k].velocity.y = vy;
                sim->grid[i][j][k].velocity.z = vz;
            }
        }
    }
}

/**
 * Update fluid dynamics parameters
 * 
 * This function is for additional fluid dynamics updates beyond what's
 * handled by the main solver.
 */
void mhd_update_fluid_dynamics(MHDSimulation *sim) {
    if (!sim) return;
    
    // Most of the fluid dynamics are handled in the main solver.
    // This function can be used for additional effects or specialized behaviors.
    
    // For example, we could add a density stratification in the z-direction
    if (sim->mode == DYNAMIC) {
        double scale_height = sim->grid_size_z / 4.0;
        double base_density = sim->initial_density;
        
        for (int i = 0; i < sim->grid_size_x; i++) {
            for (int j = 0; j < sim->grid_size_y; j++) {
                for (int k = 0; k < sim->grid_size_z; k++) {
                    // Apply exponential density stratification
                    double height = (double)k / sim->grid_size_z;
                    sim->grid[i][j][k].density = base_density * exp(-height / scale_height);
                    
                    // Adjust pressure to maintain hydrostatic equilibrium
                    sim->grid[i][j][k].pressure = sim->grid[i][j][k].density * sim->grid[i][j][k].temperature;
                }
            }
        }
    }
}

/**
 * Apply a temperature gradient to the simulation
 */
void mhd_apply_temperature_gradient(MHDSimulation *sim, double gradient_x, double gradient_y, double gradient_z) {
    if (!sim) return;
    
    double base_temp = sim->initial_temperature;
    
    
    for (int i = 0; i < sim->grid_size_x; i++) {
        for (int j = 0; j < sim->grid_size_y; j++) {
            for (int k = 0; k < sim->grid_size_z; k++) {
                // Calculate normalized coordinates (-0.5 to 0.5)
                double x_norm = ((double)i / sim->grid_size_x) - 0.5;
                double y_norm = ((double)j / sim->grid_size_y) - 0.5;
                double z_norm = ((double)k / sim->grid_size_z) - 0.5;
                
                // Apply linear gradient
                double temp = base_temp * (1.0 + 
                    gradient_x * x_norm * 2.0 + 
                    gradient_y * y_norm * 2.0 + 
                    gradient_z * z_norm * 2.0);
                
                // Ensure temperature stays positive
                if (temp <= 0.001) temp = 0.001;
                
                sim->grid[i][j][k].temperature = temp;
                
                // Adjust pressure based on ideal gas law (P = ρT)
                sim->grid[i][j][k].pressure = sim->grid[i][j][k].density * temp;
            }
        }
    }
}

/**
 * Apply a pressure gradient to the simulation
 */
void mhd_apply_pressure_gradient(MHDSimulation *sim, double gradient_x, double gradient_y, double gradient_z) {
    if (!sim) return;
    
    double base_pressure = sim->initial_pressure;
    
    for (int i = 0; i < sim->grid_size_x; i++) {
        for (int j = 0; j < sim->grid_size_y; j++) {
            for (int k = 0; k < sim->grid_size_z; k++) {
                // Calculate normalized coordinates (-0.5 to 0.5)
                double x_norm = ((double)i / sim->grid_size_x) - 0.5;
                double y_norm = ((double)j / sim->grid_size_y) - 0.5;
                double z_norm = ((double)k / sim->grid_size_z) - 0.5;
                
                // Apply linear gradient
                double pressure = base_pressure * (1.0 + 
                    gradient_x * x_norm * 2.0 + 
                    gradient_y * y_norm * 2.0 + 
                    gradient_z * z_norm * 2.0);
                
                // Ensure pressure stays positive
                if (pressure <= 0.001) pressure = 0.001;
                
                sim->grid[i][j][k].pressure = pressure;
            }
        }
    }
}

/**
 * Add a vortex flow to the velocity field
 * This is an example of a more complex flow pattern that can be added
 */
void mhd_add_vortex_flow(MHDSimulation *sim, double center_x, double center_y, double center_z, double strength) {
    if (!sim) return;
    
    // Convert center coordinates to grid indices
    int cx = (int)(center_x * sim->grid_size_x);
    int cy = (int)(center_y * sim->grid_size_y);
    int cz = (int)(center_z * sim->grid_size_z);
    
    // Ensure the center is within the grid
    if (cx < 0) cx = 0;
    if (cx >= sim->grid_size_x) cx = sim->grid_size_x - 1;
    if (cy < 0) cy = 0;
    if (cy >= sim->grid_size_y) cy = sim->grid_size_y - 1;
    if (cz < 0) cz = 0;
    if (cz >= sim->grid_size_z) cz = sim->grid_size_z - 1;
    
    // Radius of influence (half the smallest dimension)
    double radius = fmin(fmin(sim->grid_size_x, sim->grid_size_y), sim->grid_size_z) / 2.0;
    
    for (int i = 0; i < sim->grid_size_x; i++) {
        for (int j = 0; j < sim->grid_size_y; j++) {
            for (int k = 0; k < sim->grid_size_z; k++) {
                // Calculate distance from center
                double dx = i - cx;
                double dy = j - cy;
                double dz = k - cz;
                double distance = sqrt(dx*dx + dy*dy + dz*dz);
                
                // Only affect cells within the radius
                if (distance < radius) {
                    // Calculate vortex factor (decreases with distance)
                    double factor = strength * (1.0 - distance / radius);
                    
                    // Create a rotational flow around each axis
                    // The cross product of (x,y,z) with each axis gives the direction
                    
                    // Rotation around x-axis: cross product with (1,0,0)
                    sim->grid[i][j][k].velocity.y += factor * dz;
                    sim->grid[i][j][k].velocity.z -= factor * dy;
                    
                    // Rotation around y-axis: cross product with (0,1,0)
                    sim->grid[i][j][k].velocity.x -= factor * dz;
                    sim->grid[i][j][k].velocity.z += factor * dx;
                    
                    // Rotation around z-axis: cross product with (0,0,1)
                    sim->grid[i][j][k].velocity.x += factor * dy;
                    sim->grid[i][j][k].velocity.y -= factor * dx;
                }
            }
        }
    }
} 