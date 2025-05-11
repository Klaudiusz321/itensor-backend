/**
 * mhd_visualization.c - Visualization and export functions for MHD simulation
 * 
 * Implements functions for exporting simulation data for visualization and analysis.
 */

#include "mhd.h"

/**
 * Export simulation data to a binary file
 * This is the most efficient format for later reloading the simulation
 */
void mhd_export_to_binary(MHDSimulation *sim, const char *filename) {
    if (!sim || !filename) return;
    
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    // Write header information
    fwrite(&sim->grid_size_x, sizeof(int), 1, fp);
    fwrite(&sim->grid_size_y, sizeof(int), 1, fp);
    fwrite(&sim->grid_size_z, sizeof(int), 1, fp);
    fwrite(&sim->time_step, sizeof(double), 1, fp);
    fwrite(&sim->current_time, sizeof(double), 1, fp);
    fwrite(&sim->total_time, sizeof(double), 1, fp);
    
    // Write simulation parameters
    int method = (int)sim->method;
    int boundary_type = (int)sim->boundary_type;
    int mode = (int)sim->mode;
    
    fwrite(&method, sizeof(int), 1, fp);
    fwrite(&boundary_type, sizeof(int), 1, fp);
    fwrite(&mode, sizeof(int), 1, fp);
    
    fwrite(&sim->initial_density, sizeof(double), 1, fp);
    fwrite(&sim->initial_pressure, sizeof(double), 1, fp);
    fwrite(&sim->initial_temperature, sizeof(double), 1, fp);
    
    fwrite(&sim->initial_velocity, sizeof(Vector3D), 1, fp);
    fwrite(&sim->initial_magnetic_field, sizeof(Vector3D), 1, fp);
    
    fwrite(&sim->magnetic_conductivity, sizeof(double), 1, fp);
    fwrite(&sim->magnetic_viscosity, sizeof(double), 1, fp);
    
    fwrite(&sim->turbulence_intensity, sizeof(double), 1, fp);
    fwrite(&sim->use_particle_interaction, sizeof(bool), 1, fp);
    fwrite(&sim->use_spatial_gradients, sizeof(bool), 1, fp);
    fwrite(&sim->random_disturbance_intensity, sizeof(double), 1, fp);
    
    // Write grid data
    for (int i = 0; i < sim->grid_size_x; i++) {
        for (int j = 0; j < sim->grid_size_y; j++) {
            fwrite(sim->grid[i][j], sizeof(GridCell), sim->grid_size_z, fp);
        }
    }
    
    fclose(fp);
    printf("Simulation exported to binary file: %s\n", filename);
}

/**
 * Export a specific field (density, pressure, etc.) to a CSV file
 */
void mhd_export_field_data(MHDSimulation *sim, const char *field_name, const char *filename) {
    if (!sim || !field_name || !filename) return;
    
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    // Write CSV header (i,j,k,value)
    fprintf(fp, "i,j,k,%s\n", field_name);
    
    // Export the requested field
    for (int i = 0; i < sim->grid_size_x; i++) {
        for (int j = 0; j < sim->grid_size_y; j++) {
            for (int k = 0; k < sim->grid_size_z; k++) {
                if (strcmp(field_name, "density") == 0) {
                    fprintf(fp, "%d,%d,%d,%g\n", i, j, k, sim->grid[i][j][k].density);
                }
                else if (strcmp(field_name, "pressure") == 0) {
                    fprintf(fp, "%d,%d,%d,%g\n", i, j, k, sim->grid[i][j][k].pressure);
                }
                else if (strcmp(field_name, "temperature") == 0) {
                    fprintf(fp, "%d,%d,%d,%g\n", i, j, k, sim->grid[i][j][k].temperature);
                }
                else if (strcmp(field_name, "velocity_x") == 0) {
                    fprintf(fp, "%d,%d,%d,%g\n", i, j, k, sim->grid[i][j][k].velocity.x);
                }
                else if (strcmp(field_name, "velocity_y") == 0) {
                    fprintf(fp, "%d,%d,%d,%g\n", i, j, k, sim->grid[i][j][k].velocity.y);
                }
                else if (strcmp(field_name, "velocity_z") == 0) {
                    fprintf(fp, "%d,%d,%d,%g\n", i, j, k, sim->grid[i][j][k].velocity.z);
                }
                else if (strcmp(field_name, "velocity_magnitude") == 0) {
                    Vector3D v = sim->grid[i][j][k].velocity;
                    double magnitude = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
                    fprintf(fp, "%d,%d,%d,%g\n", i, j, k, magnitude);
                }
                else if (strcmp(field_name, "magnetic_x") == 0) {
                    fprintf(fp, "%d,%d,%d,%g\n", i, j, k, sim->grid[i][j][k].magnetic.x);
                }
                else if (strcmp(field_name, "magnetic_y") == 0) {
                    fprintf(fp, "%d,%d,%d,%g\n", i, j, k, sim->grid[i][j][k].magnetic.y);
                }
                else if (strcmp(field_name, "magnetic_z") == 0) {
                    fprintf(fp, "%d,%d,%d,%g\n", i, j, k, sim->grid[i][j][k].magnetic.z);
                }
                else if (strcmp(field_name, "magnetic_magnitude") == 0) {
                    Vector3D b = sim->grid[i][j][k].magnetic;
                    double magnitude = sqrt(b.x*b.x + b.y*b.y + b.z*b.z);
                    fprintf(fp, "%d,%d,%d,%g\n", i, j, k, magnitude);
                }
                else {
                    fprintf(stderr, "Warning: Unknown field '%s', using density\n", field_name);
                    fprintf(fp, "%d,%d,%d,%g\n", i, j, k, sim->grid[i][j][k].density);
                }
            }
        }
    }
    
    fclose(fp);
    printf("Field '%s' exported to CSV file: %s\n", field_name, filename);
}

/**
 * Export a 2D slice of the simulation data to a CSV file
 */
void mhd_export_slice_data(MHDSimulation *sim, int slice_dim, int slice_pos, const char *filename) {
    if (!sim || !filename) return;
    
    // Check slice dimension (0=x, 1=y, 2=z)
    if (slice_dim < 0 || slice_dim > 2) {
        fprintf(stderr, "Error: Invalid slice dimension %d (must be 0, 1, or 2)\n", slice_dim);
        return;
    }
    
    // Check slice position
    int max_pos;
    if (slice_dim == 0) {
        max_pos = sim->grid_size_x;
    } else if (slice_dim == 1) {
        max_pos = sim->grid_size_y;
    } else {
        max_pos = sim->grid_size_z;
    }
    
    if (slice_pos < 0 || slice_pos >= max_pos) {
        fprintf(stderr, "Error: Invalid slice position %d (must be between 0 and %d)\n", 
                slice_pos, max_pos - 1);
        return;
    }
    
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    // Write CSV header
    if (slice_dim == 0) {
        fprintf(fp, "j,k,density,pressure,temperature,velocity_x,velocity_y,velocity_z,magnetic_x,magnetic_y,magnetic_z\n");
    } else if (slice_dim == 1) {
        fprintf(fp, "i,k,density,pressure,temperature,velocity_x,velocity_y,velocity_z,magnetic_x,magnetic_y,magnetic_z\n");
    } else {
        fprintf(fp, "i,j,density,pressure,temperature,velocity_x,velocity_y,velocity_z,magnetic_x,magnetic_y,magnetic_z\n");
    }
    
    // Export the slice
    if (slice_dim == 0) {  // x-slice (constant i)
        for (int j = 0; j < sim->grid_size_y; j++) {
            for (int k = 0; k < sim->grid_size_z; k++) {
                GridCell *cell = &sim->grid[slice_pos][j][k];
                fprintf(fp, "%d,%d,%g,%g,%g,%g,%g,%g,%g,%g,%g\n",
                        j, k, 
                        cell->density, cell->pressure, cell->temperature,
                        cell->velocity.x, cell->velocity.y, cell->velocity.z,
                        cell->magnetic.x, cell->magnetic.y, cell->magnetic.z);
            }
        }
    } else if (slice_dim == 1) {  // y-slice (constant j)
        for (int i = 0; i < sim->grid_size_x; i++) {
            for (int k = 0; k < sim->grid_size_z; k++) {
                GridCell *cell = &sim->grid[i][slice_pos][k];
                fprintf(fp, "%d,%d,%g,%g,%g,%g,%g,%g,%g,%g,%g\n",
                        i, k, 
                        cell->density, cell->pressure, cell->temperature,
                        cell->velocity.x, cell->velocity.y, cell->velocity.z,
                        cell->magnetic.x, cell->magnetic.y, cell->magnetic.z);
            }
        }
    } else {  // z-slice (constant k)
        for (int i = 0; i < sim->grid_size_x; i++) {
            for (int j = 0; j < sim->grid_size_y; j++) {
                GridCell *cell = &sim->grid[i][j][slice_pos];
                fprintf(fp, "%d,%d,%g,%g,%g,%g,%g,%g,%g,%g,%g\n",
                        i, j, 
                        cell->density, cell->pressure, cell->temperature,
                        cell->velocity.x, cell->velocity.y, cell->velocity.z,
                        cell->magnetic.x, cell->magnetic.y, cell->magnetic.z);
            }
        }
    }
    
    fclose(fp);
    printf("Slice data exported to CSV file: %s\n", filename);
}

/**
 * Export data in HDF5 format (just a placeholder - would require HDF5 library)
 */
void mhd_export_to_hdf5(MHDSimulation *sim, const char *filename) {
    if (!sim || !filename) return;
    
    // This is just a placeholder function. In a real implementation,
    // you would need to link against the HDF5 library and use its API.
    
    printf("HDF5 export not implemented. Would export to: %s\n", filename);
    printf("To implement HDF5 export, add HDF5 library dependencies and implement this function.\n");
    
    // Example of what the implementation would look like:
    /*
    hid_t file_id, dataset_id, dataspace_id;
    hsize_t dims[3];
    herr_t status;
    
    dims[0] = sim->grid_size_x;
    dims[1] = sim->grid_size_y;
    dims[2] = sim->grid_size_z;
    
    // Create HDF5 file
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    // Create dataspace for the dataset
    dataspace_id = H5Screate_simple(3, dims, NULL);
    
    // Create a dataset for each field
    // ... (density, pressure, velocity, etc.)
    
    // Close resources
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    */
} 