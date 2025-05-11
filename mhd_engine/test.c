#include <stdio.h>
#include "mhd.h"


int main() {
    // Create a smaller simulation for faster testing
    MHDSimulation *sim = mhd_initialize(30, 30, 30);
    
    // Setup the simulation
    mhd_initialize_magnetic_field(sim, 0.5, 0.5, 0.0);
    mhd_initialize_fluid_parameters(sim, 1.0, 1.0, 1.0);
    mhd_set_initial_velocity(sim, 0.1, 0.0, 0.0);
    mhd_set_boundary_type(sim, PERIODIC);
    mhd_set_numerical_method(sim, RUNGE_KUTTA_2);
    
    // Add some interesting features
    mhd_apply_turbulence(sim, 0.3);
    mhd_apply_spatial_gradients(sim, true);
    
    // Run for a short time
    mhd_set_total_time(sim, 0.5);
    mhd_set_time_step(sim, 0.01);
    printf("Running custom MHD test...\n");
    mhd_run_simulation(sim);
    
    // Export different views of the data
    mhd_export_field_data(sim, "density",           "density.csv");
    mhd_export_field_data(sim, "pressure",          "pressure.csv");
    mhd_export_field_data(sim, "temperature",       "temperature.csv");
    mhd_export_field_data(sim, "velocity_magnitude","velocity.csv");
    mhd_export_field_data(sim, "magnetic_magnitude","magnetic.csv");

    int cx = sim->grid_size_x / 2;
    int cy = sim->grid_size_y / 2;
    int cz = sim->grid_size_z / 2;
    mhd_export_slice_data(sim, 0, cx, "x_slice.csv");
    mhd_export_slice_data(sim, 1, cy, "y_slice.csv");
    mhd_export_slice_data(sim, 2, cz, "z_slice.csv");
    
    mhd_free(sim);
    printf("Custom test completed!\n");
    return 0;
}