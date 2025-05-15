/**
 * mhd_magnetic.c - Magnetic field functions for MHD simulation
 * 
 * Implements functions for managing magnetic field parameters and effects
 * in the MHD simulation.
 */

#include "mhd.h"

/**
 * Initialize the magnetic field with uniform values
 */
void mhd_initialize_magnetic_field(MHDSimulation *sim, double bx, double by, double bz) {
    if (!sim) return;
    
    fprintf(stderr, "Initializing magnetic field: B=(%g, %g, %g)\n", bx, by, bz);
    
    // Zapisz wartości w strukturze symulacji
    sim->initial_magnetic_field.x = bx;
    sim->initial_magnetic_field.y = by;
    sim->initial_magnetic_field.z = bz;
    
    // Ustaw pole magnetyczne we wszystkich komórkach siatki
    if (!sim->grid) {
        fprintf(stderr, "Warning: Grid not allocated in mhd_initialize_magnetic_field\n");
        return;
    }
    
    for (int i = 0; i < sim->grid_size_x; i++) {
        for (int j = 0; j < sim->grid_size_y; j++) {
            for (int k = 0; k < sim->grid_size_z; k++) {
                sim->grid[i][j][k].magnetic.x = bx;
                sim->grid[i][j][k].magnetic.y = by;
                sim->grid[i][j][k].magnetic.z = bz;
            }
        }
    }
    
    fprintf(stderr, "Magnetic field initialized successfully\n");
}

/**
 * Set the magnetic conductivity parameter
 */
void mhd_set_magnetic_conductivity(MHDSimulation *sim, double conductivity) {
    if (!sim) return;
    
    if (conductivity < 0.0) {
        fprintf(stderr, "Warning: Magnetic conductivity cannot be negative, using absolute value\n");
        conductivity = fabs(conductivity);
    }
    
    sim->magnetic_conductivity = conductivity;
}

/**
 * Set the magnetic viscosity parameter
 */
void mhd_set_magnetic_viscosity(MHDSimulation *sim, double viscosity) {
    if (!sim) return;
    
    if (viscosity < 0.0) {
        fprintf(stderr, "Warning: Magnetic viscosity cannot be negative, using absolute value\n");
        viscosity = fabs(viscosity);
    }
    
    sim->magnetic_viscosity = viscosity;
}

/**
 * Update the magnetic field using the MHD induction equation
 * 
 * This function applies the magnetic field changes that happen during a timestep.
 * It's separate from the main solver to allow for specific magnetic field effects.
 */
void mhd_update_magnetic_field(MHDSimulation *sim) {
    if (!sim) return;
    
    // Most of the magnetic field update is handled in the main solver
    // through the MHD equations. This function is for additional effects
    // or specialized magnetic field behaviors.
    
    // For example, we could add a pulsating magnetic field for dynamic simulation
    if (sim->mode == MODE_DYNAMIC) {
        double frequency = 0.1; // Frequency of pulsation
        double amplitude = 0.2; // Amplitude of pulsation
        
        // Calculate pulsation factor based on current time
        double pulsation = 1.0 + amplitude * sin(2.0 * PI * frequency * sim->current_time);
        
        // Apply pulsation to the magnetic field
        for (int i = 1; i < sim->grid_size_x - 1; i++) {
            for (int j = 1; j < sim->grid_size_y - 1; j++) {
                for (int k = 1; k < sim->grid_size_z - 1; k++) {
                    sim->grid[i][j][k].magnetic.x = sim->initial_magnetic_field.x * pulsation;
                    sim->grid[i][j][k].magnetic.y = sim->initial_magnetic_field.y * pulsation;
                    sim->grid[i][j][k].magnetic.z = sim->initial_magnetic_field.z * pulsation;
                }
            }
        }
    }
}

/**
 * Apply magnetic viscosity effects
 * 
 * Magnetic viscosity is the diffusion of the magnetic field over time,
 * similar to how regular viscosity causes momentum to diffuse in a fluid.
 */
void mhd_apply_magnetic_viscosity(MHDSimulation *sim) {
    if (!sim || sim->magnetic_viscosity <= 0.0) return;
    
    // Magnetic viscosity effects are already included in the main solver
    // through the η∇²B term in the induction equation.
    // This function can be used for additional viscosity effects or diagnostics.
    
    // For demonstration, we can calculate the total magnetic energy dissipation due to viscosity
    double total_dissipation = 0.0;
    
    for (int i = 1; i < sim->grid_size_x - 1; i++) {
        for (int j = 1; j < sim->grid_size_y - 1; j++) {
            for (int k = 1; k < sim->grid_size_z - 1; k++) {
                GridCell *cell = &sim->grid[i][j][k];
                GridCell *cell_x_plus = &sim->grid[i+1][j][k];
                GridCell *cell_x_minus = &sim->grid[i-1][j][k];
                GridCell *cell_y_plus = &sim->grid[i][j+1][k];
                GridCell *cell_y_minus = &sim->grid[i][j-1][k];
                GridCell *cell_z_plus = &sim->grid[i][j][k+1];
                GridCell *cell_z_minus = &sim->grid[i][j][k-1];
                
                // Calculate the Laplacian of the magnetic field (∇²B)
                double laplacian_Bx = 
                    (cell_x_plus->magnetic.x + cell_x_minus->magnetic.x +
                     cell_y_plus->magnetic.x + cell_y_minus->magnetic.x +
                     cell_z_plus->magnetic.x + cell_z_minus->magnetic.x - 6 * cell->magnetic.x);
                
                double laplacian_By = 
                    (cell_x_plus->magnetic.y + cell_x_minus->magnetic.y +
                     cell_y_plus->magnetic.y + cell_y_minus->magnetic.y +
                     cell_z_plus->magnetic.y + cell_z_minus->magnetic.y - 6 * cell->magnetic.y);
                
                double laplacian_Bz = 
                    (cell_x_plus->magnetic.z + cell_x_minus->magnetic.z +
                     cell_y_plus->magnetic.z + cell_y_minus->magnetic.z +
                     cell_z_plus->magnetic.z + cell_z_minus->magnetic.z - 6 * cell->magnetic.z);
                
                // Calculate the magnetic energy dissipation rate: η|∇²B|²
                double dissipation_rate = sim->magnetic_viscosity * (
                    laplacian_Bx * laplacian_Bx +
                    laplacian_By * laplacian_By +
                    laplacian_Bz * laplacian_Bz
                );
                
                total_dissipation += dissipation_rate;
            }
        }
    }
    
    // Optional: We could use this to track energy conservation in the simulation
    // printf("Total magnetic energy dissipation rate: %f\n", total_dissipation);
}

/**
 * Apply magnetic conductivity effects
 * 
 * Magnetic conductivity affects how strongly the fluid and magnetic field interact,
 * specifically through the Lorentz force in the momentum equation.
 */
void mhd_apply_magnetic_conductivity(MHDSimulation *sim) {
    if (!sim || sim->magnetic_conductivity <= 0.0) return;
    
    // Magnetic conductivity effects are already included in the main solver
    // through the j×B term in the momentum equation, scaled by conductivity.
    // This function can be used for additional conductivity effects or diagnostics.
    
    // For demonstration, we can calculate the total Lorentz force in the system
    double total_lorentz_force = 0.0;
    
    for (int i = 1; i < sim->grid_size_x - 1; i++) {
        for (int j = 1; j < sim->grid_size_y - 1; j++) {
            for (int k = 1; k < sim->grid_size_z - 1; k++) {
                GridCell *cell = &sim->grid[i][j][k];
                GridCell *cell_x_plus = &sim->grid[i+1][j][k];
                GridCell *cell_x_minus = &sim->grid[i-1][j][k];
                GridCell *cell_y_plus = &sim->grid[i][j+1][k];
                GridCell *cell_y_minus = &sim->grid[i][j-1][k];
                GridCell *cell_z_plus = &sim->grid[i][j][k+1];
                GridCell *cell_z_minus = &sim->grid[i][j][k-1];
                
                // Calculate curl of B (∇×B) to get current density j
                double curl_B_x = 
                    (cell_y_plus->magnetic.z - cell_y_minus->magnetic.z) / 2.0 -
                    (cell_z_plus->magnetic.y - cell_z_minus->magnetic.y) / 2.0;
                
                double curl_B_y = 
                    (cell_z_plus->magnetic.x - cell_z_minus->magnetic.x) / 2.0 -
                    (cell_x_plus->magnetic.z - cell_x_minus->magnetic.z) / 2.0;
                
                double curl_B_z = 
                    (cell_x_plus->magnetic.y - cell_x_minus->magnetic.y) / 2.0 -
                    (cell_y_plus->magnetic.x - cell_y_minus->magnetic.x) / 2.0;
                
                // Calculate Lorentz force j×B
                double lorentz_x = curl_B_y * cell->magnetic.z - curl_B_z * cell->magnetic.y;
                double lorentz_y = curl_B_z * cell->magnetic.x - curl_B_x * cell->magnetic.z;
                double lorentz_z = curl_B_x * cell->magnetic.y - curl_B_y * cell->magnetic.x;
                
                // Scale by conductivity
                lorentz_x *= sim->magnetic_conductivity;
                lorentz_y *= sim->magnetic_conductivity;
                lorentz_z *= sim->magnetic_conductivity;
                
                // Calculate magnitude of Lorentz force
                double lorentz_magnitude = sqrt(
                    lorentz_x * lorentz_x +
                    lorentz_y * lorentz_y +
                    lorentz_z * lorentz_z
                );
                
                total_lorentz_force += lorentz_magnitude;
            }
        }
    }
    
    // Optional: We could use this to track the influence of magnetic forces in the simulation
    // printf("Total Lorentz force magnitude: %f\n", total_lorentz_force);
} 