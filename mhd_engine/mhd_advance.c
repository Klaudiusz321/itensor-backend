/**
 * mhd_advance.c - Advanced simulation options for MHD simulation
 * 
 * Implements specialized and advanced features for MHD simulations including
 * turbulence, disturbances, particle interactions, and spatial gradients.
 */

#include "mhd.h"

/**
 * Set the simulation mode
 */
void mhd_set_simulation_mode(MHDSimulation *sim, SimulationMode mode) {
    if (!sim) return;
    sim->mode = mode;
}

/**
 * Apply random disturbances to the simulation
 * 
 * This adds random perturbations to all fields, with intensity controlling
 * the magnitude of the disturbances.
 */
void mhd_apply_disturbances(MHDSimulation *sim, double intensity) {
    if (!sim || intensity <= 0.0) return;
    
    // Set the disturbance intensity for future use
    sim->random_disturbance_intensity = intensity;
    
    // Apply random disturbances to the grid
    mhd_add_random_disturbance(sim, intensity);
}

/**
 * Apply dynamic changes to the simulation
 * 
 * This function handles time-dependent changes for dynamic simulations,
 * such as oscillating fields or evolving parameters.
 */
void mhd_apply_dynamic_changes(MHDSimulation *sim) {
    if (!sim) return;
    
    // Example: Oscillating velocity field
    double time = sim->current_time;
    double frequency = 0.2;  // Frequency of oscillation
    double amplitude = 0.1;  // Amplitude of oscillation
    
    // Calculate oscillation factor
    double oscillation = amplitude * sin(2.0 * PI * frequency * time);
    
    // Apply to the entire grid
    for (int i = 1; i < sim->grid_size_x - 1; i++) {
        for (int j = 1; j < sim->grid_size_y - 1; j++) {
            for (int k = 1; k < sim->grid_size_z - 1; k++) {
                // Oscillate the x-velocity component
                sim->grid[i][j][k].velocity.x += oscillation;
                
                // Example of rotating oscillation: 
                // Different phase in different directions
                sim->grid[i][j][k].velocity.y += amplitude * cos(2.0 * PI * frequency * time);
                sim->grid[i][j][k].velocity.z += amplitude * sin(2.0 * PI * frequency * time + PI/4);
            }
        }
    }
    
    // Other dynamic changes could include:
    // - Time-dependent magnetic field
    // - Evolving density structure
    // - Progressive heating/cooling
}

/**
 * Apply turbulence to the velocity field
 * 
 * This adds turbulent flow structures to the simulation.
 */
void mhd_apply_turbulence(MHDSimulation *sim, double intensity) {
    if (!sim || intensity <= 0.0) return;
    
    // Set the turbulence intensity for future use
    sim->turbulence_intensity = intensity;
    
    // We'll use several vortices of different scales to create turbulence
    // For simplicity, we're adding a few random vortices
    
    // Number of vortices scales with turbulence intensity
    int num_vortices = (int)(5.0 * intensity);
    if (num_vortices < 1) num_vortices = 1;
    
    for (int v = 0; v < num_vortices; v++) {
        // Random position for the vortex center
        double center_x = mhd_random_value(0.1, 0.9);
        double center_y = mhd_random_value(0.1, 0.9);
        double center_z = mhd_random_value(0.1, 0.9);
        
        // Random strength, scaled by turbulence intensity
        double strength = mhd_random_value(0.01, 0.1) * intensity;
        
        // Add the vortex to the flow
        for (int i = 1; i < sim->grid_size_x - 1; i++) {
            for (int j = 1; j < sim->grid_size_y - 1; j++) {
                for (int k = 1; k < sim->grid_size_z - 1; k++) {
                    // Calculate normalized coordinates
                    double x_norm = (double)i / sim->grid_size_x;
                    double y_norm = (double)j / sim->grid_size_y;
                    double z_norm = (double)k / sim->grid_size_z;
                    
                    // Calculate distance from vortex center
                    double dx = x_norm - center_x;
                    double dy = y_norm - center_y;
                    double dz = z_norm - center_z;
                    double distance = sqrt(dx*dx + dy*dy + dz*dz);
                    
                    // Vortex radius (smaller vortices for turbulence)
                    double radius = mhd_random_value(0.05, 0.2);
                    
                    // Only apply within the vortex radius
                    if (distance < radius) {
                        // Calculate velocity change (curl-like)
                        // The factor ensures velocity goes to zero at the edge
                        double factor = strength * (1.0 - distance / radius);
                        
                        // Add a rotational component around a random axis
                        // We'll use different axes for different vortices to create turbulence
                        switch (v % 3) {
                            case 0: // Rotation around z
                                sim->grid[i][j][k].velocity.x += factor * dy * 10.0;
                                sim->grid[i][j][k].velocity.y -= factor * dx * 10.0;
                                break;
                            case 1: // Rotation around x
                                sim->grid[i][j][k].velocity.y += factor * dz * 10.0;
                                sim->grid[i][j][k].velocity.z -= factor * dy * 10.0;
                                break;
                            case 2: // Rotation around y
                                sim->grid[i][j][k].velocity.x -= factor * dz * 10.0;
                                sim->grid[i][j][k].velocity.z += factor * dx * 10.0;
                                break;
                        }
                    }
                }
            }
        }
    }
}

/**
 * Apply particle interaction effects
 * 
 * This simulates the interaction of charged particles with the magnetic field.
 */
void mhd_apply_particle_interaction(MHDSimulation *sim, bool enable) {
    if (!sim) return;
    
    sim->use_particle_interaction = enable;
    
    if (!enable) return;
    
    // This is a simplified model where we add an additional term to the 
    // velocity field based on the Lorentz force on charged particles
    
    for (int i = 1; i < sim->grid_size_x - 1; i++) {
        for (int j = 1; j < sim->grid_size_y - 1; j++) {
            for (int k = 1; k < sim->grid_size_z - 1; k++) {
                GridCell *cell = &sim->grid[i][j][k];
                
                // Calculate Lorentz force: F = q(E + v×B)
                // We ignore the electric field E for simplicity
                double lorentz_x = cell->velocity.y * cell->magnetic.z - cell->velocity.z * cell->magnetic.y;
                double lorentz_y = cell->velocity.z * cell->magnetic.x - cell->velocity.x * cell->magnetic.z;
                double lorentz_z = cell->velocity.x * cell->magnetic.y - cell->velocity.y * cell->magnetic.x;
                
                // Scale factor represents the charge-to-mass ratio and time step
                double scale = 0.01;
                
                // Update velocity due to Lorentz force (simplified)
                cell->velocity.x += scale * lorentz_x;
                cell->velocity.y += scale * lorentz_y;
                cell->velocity.z += scale * lorentz_z;
            }
        }
    }
}

/**
 * Apply spatial gradients to the simulation
 * 
 * This allows creating structured spatial variations in the fields.
 */
void mhd_apply_spatial_gradients(MHDSimulation *sim, bool enable) {
    if (!sim) return;
    
    sim->use_spatial_gradients = enable;
    
    if (!enable) return;
    
    // Apply various gradients to create a more complex initial state
    
    // Apply a density gradient along the x-axis
    double density_gradient_x = 0.5;  // Increase by 50% across the domain
    double base_density = sim->initial_density;
    
    // Apply a temperature gradient along the z-axis
    double temp_gradient_z = 1.0;     // Double across the domain
    double base_temp = sim->initial_temperature;
    
    // Apply a magnetic field gradient along the y-axis
    double magnetic_gradient_y = 0.3; // Increase by 30% across the domain
    double base_magnetic_x = sim->initial_magnetic_field.x;
    double base_magnetic_y = sim->initial_magnetic_field.y;
    double base_magnetic_z = sim->initial_magnetic_field.z;
    
    for (int i = 0; i < sim->grid_size_x; i++) {
        for (int j = 0; j < sim->grid_size_y; j++) {
            for (int k = 0; k < sim->grid_size_z; k++) {
                // Calculate normalized coordinates (0 to 1)
                double x_norm = (double)i / sim->grid_size_x;
                double y_norm = (double)j / sim->grid_size_y;
                double z_norm = (double)k / sim->grid_size_z;
                
                // Apply density gradient along x
                sim->grid[i][j][k].density = base_density * (1.0 + density_gradient_x * x_norm);
                
                // Apply temperature gradient along z
                sim->grid[i][j][k].temperature = base_temp * (1.0 + temp_gradient_z * z_norm);
                
                // Update pressure based on new density and temperature
                sim->grid[i][j][k].pressure = sim->grid[i][j][k].density * sim->grid[i][j][k].temperature;
                
                // Apply magnetic field gradient along y
                sim->grid[i][j][k].magnetic.x = base_magnetic_x * (1.0 + magnetic_gradient_y * y_norm);
                sim->grid[i][j][k].magnetic.y = base_magnetic_y * (1.0 + magnetic_gradient_y * y_norm);
                sim->grid[i][j][k].magnetic.z = base_magnetic_z * (1.0 + magnetic_gradient_y * y_norm);
            }
        }
    }
}

/**
 * Set the intensity of random disturbances
 */
void mhd_set_random_disturbance_intensity(MHDSimulation *sim, double intensity) {
    if (!sim) return;
    
    if (intensity < 0.0) {
        fprintf(stderr, "Warning: Disturbance intensity cannot be negative, using absolute value\n");
        intensity = fabs(intensity);
    }
    
    sim->random_disturbance_intensity = intensity;
} 