#include <stdio.h>
#include "mhd.h"
#include <math.h>
#include <string.h>
#include <ctype.h>

// Helper function to trim whitespace from a string
char* trim_string(char* str) {
    if (!str) return NULL;
    
    // Trim leading space
    while(isspace((unsigned char)*str)) str++;
    
    if(*str == 0)  // All spaces?
        return str;
    
    // Trim trailing space
    char* end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;
    
    // Write new null terminator
    *(end+1) = 0;
    
    return str;
}

// Case insensitive string comparison
int strcicmp(const char *a, const char *b) {
    if (!a || !b) return a == b ? 0 : (a ? 1 : -1);
    
    for (;; a++, b++) {
        int d = tolower((unsigned char)*a) - tolower((unsigned char)*b);
        if (d != 0 || !*a)
            return d;
    }
}

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
 * This is a simplified version that generates sample data instead of accessing the grid
 */
void mhd_export_field_data(MHDSimulation *sim,
                           const char    *field_name,
                           const char    *filename)
{
    if (!sim || !field_name || !filename) return;

    // przygotuj nazwę pola (trim + lowercase jeśli potrzebne)
    char field_copy[64];
    strncpy(field_copy, field_name, sizeof(field_copy)-1);
    field_copy[sizeof(field_copy)-1] = '\0';
    char *fn = trim_string(field_copy);

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Cannot open %s for writing\n", filename);
        return;
    }

    // Specjalne przetwarzanie dla pól magnitude
    double **velocity_magnitude = NULL;
    double **magnetic_magnitude = NULL;
    
    // Jeśli potrzebujemy pól magnitude, obliczamy je
    if (strcmp(fn, "velocity_magnitude") == 0 || strcmp(fn, "magnetic_magnitude") == 0) {
        mhd_calculate_vector_magnitudes(sim, &velocity_magnitude, &magnetic_magnitude);
        
        // Sprawdź czy udało się zaalokować pamięć
        if ((strcmp(fn, "velocity_magnitude") == 0 && !velocity_magnitude) ||
            (strcmp(fn, "magnetic_magnitude") == 0 && !magnetic_magnitude)) {
            fprintf(stderr, "Failed to calculate vector magnitudes\n");
            fclose(fp);
            return;
        }
    }

    // Zdecyduj nagłówek CSV:
    if (strcmp(fn, "velocity") == 0 ||
        strcmp(fn, "velocity_field") == 0 ||
        strcmp(fn, "magnetic_field") == 0 ||
        strcmp(fn, "magnetic_vector") == 0)
    {
        // wszystkie wektorowe pola mają kolumny vx,vy,vz
        fprintf(fp, "i,j,k,vx,vy,vz\n");
    } else {
        // skalary
        fprintf(fp, "i,j,k,value\n");
    }

    // Pętla po komórkach
    for (int i = 0; i < sim->grid_size_x; ++i) {
      for (int j = 0; j < sim->grid_size_y; ++j) {
        for (int k = 0; k < sim->grid_size_z; ++k) {
          GridCell *c = &sim->grid[i][j][k];

          if (strcmp(fn, "velocity") == 0 ||
              strcmp(fn, "velocity_field") == 0)
          {
            // wektor prędkości
            fprintf(fp, "%d,%d,%d,%.10g,%.10g,%.10g\n",
                    i, j, k,
                    c->velocity.x,
                    c->velocity.y,
                    c->velocity.z);
          }
          else if (strcmp(fn, "magnetic_field") == 0 ||
                   strcmp(fn, "magnetic_vector") == 0)
          {
            // wektor pola magnetycznego
            fprintf(fp, "%d,%d,%d,%.10g,%.10g,%.10g\n",
                    i, j, k,
                    c->magnetic.x,
                    c->magnetic.y,
                    c->magnetic.z);
          }
          else if (strcmp(fn, "velocity_magnitude") == 0) {
            // Wielkość prędkości (tylko dla k=0, bo to 2D)
            if (k == 0 && velocity_magnitude) {
                fprintf(fp, "%d,%d,%d,%.10g\n", i, j, k, velocity_magnitude[i][j]);
            }
          }
          else if (strcmp(fn, "magnetic_magnitude") == 0) {
            // Wielkość pola magnetycznego (tylko dla k=0, bo to 2D)
            if (k == 0 && magnetic_magnitude) {
                fprintf(fp, "%d,%d,%d,%.10g\n", i, j, k, magnetic_magnitude[i][j]);
            }
          }
          else if (strcmp(fn, "density") == 0) {
            fprintf(fp, "%d,%d,%d,%.10g\n", i, j, k, c->density);
          }
          else if (strcmp(fn, "pressure") == 0) {
            fprintf(fp, "%d,%d,%d,%.10g\n", i, j, k, c->pressure);
          }
          else if (strcmp(fn, "temperature") == 0) {
            fprintf(fp, "%d,%d,%d,%.10g\n", i, j, k, c->temperature);
          }
          // jeżeli masz inne pola scalarne, dopisz je analogicznie
        }
      }
    }

    // Zwolnij pamięć po obliczonych polach
    if (velocity_magnitude || magnetic_magnitude) {
        mhd_free_vector_magnitudes(&velocity_magnitude, &magnetic_magnitude, sim->grid_size_x);
    }

    fclose(fp);
}


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
 * Calculate magnitude of vector fields and store them in separate arrays
 * This function computes the magnitude of velocity and magnetic field vectors
 * and stores the results in 2D arrays for visualization and analysis
 */
void mhd_calculate_vector_magnitudes(MHDSimulation *sim, 
                                    double ***velocity_magnitude, 
                                    double ***magnetic_magnitude) 
{
    if (!sim || !velocity_magnitude || !magnetic_magnitude) return;
    
    int nx = sim->grid_size_x;
    int ny = sim->grid_size_y;
    int nz = sim->grid_size_z;
    
    // Allocate memory for magnitude arrays if not already allocated
    if (*velocity_magnitude == NULL) {
        *velocity_magnitude = (double**)malloc(nx * sizeof(double*));
        if (!(*velocity_magnitude)) {
            fprintf(stderr, "Error: Failed to allocate memory for velocity magnitude\n");
            return;
        }
        
        for (int i = 0; i < nx; i++) {
            (*velocity_magnitude)[i] = (double*)malloc(ny * sizeof(double));
            if (!(*velocity_magnitude)[i]) {
                fprintf(stderr, "Error: Failed to allocate memory for velocity magnitude row %d\n", i);
                // Clean up previously allocated memory
                for (int j = 0; j < i; j++) {
                    free((*velocity_magnitude)[j]);
                }
                free(*velocity_magnitude);
                *velocity_magnitude = NULL;
                return;
            }
        }
    }
    
    if (*magnetic_magnitude == NULL) {
        *magnetic_magnitude = (double**)malloc(nx * sizeof(double*));
        if (!(*magnetic_magnitude)) {
            fprintf(stderr, "Error: Failed to allocate memory for magnetic magnitude\n");
            return;
        }
        
        for (int i = 0; i < nx; i++) {
            (*magnetic_magnitude)[i] = (double*)malloc(ny * sizeof(double));
            if (!(*magnetic_magnitude)[i]) {
                fprintf(stderr, "Error: Failed to allocate memory for magnetic magnitude row %d\n", i);
                // Clean up previously allocated memory
                for (int j = 0; j < i; j++) {
                    free((*magnetic_magnitude)[j]);
                }
                free(*magnetic_magnitude);
                *magnetic_magnitude = NULL;
                return;
            }
        }
    }
    
    // Calculate magnitudes for each cell
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            // We use only the first layer in z-direction for 2D visualization
            int k = 0;
            
            // Calculate velocity magnitude: sqrt(vx^2 + vy^2 + vz^2)
            double vx = sim->grid[i][j][k].velocity.x;
            double vy = sim->grid[i][j][k].velocity.y;
            double vz = sim->grid[i][j][k].velocity.z;
            (*velocity_magnitude)[i][j] = sqrt(vx*vx + vy*vy + vz*vz);
            
            // Calculate magnetic field magnitude: sqrt(Bx^2 + By^2 + Bz^2)
            double Bx = sim->grid[i][j][k].magnetic.x;
            double By = sim->grid[i][j][k].magnetic.y;
            double Bz = sim->grid[i][j][k].magnetic.z;
            (*magnetic_magnitude)[i][j] = sqrt(Bx*Bx + By*By + Bz*Bz);
        }
    }
}

/**
 * Free memory allocated for vector magnitude arrays
 */
void mhd_free_vector_magnitudes(double ***velocity_magnitude, 
                               double ***magnetic_magnitude,
                               int nx) 
{
    if (velocity_magnitude && *velocity_magnitude) {
        for (int i = 0; i < nx; i++) {
            if ((*velocity_magnitude)[i]) {
                free((*velocity_magnitude)[i]);
            }
        }
        free(*velocity_magnitude);
        *velocity_magnitude = NULL;
    }
    
    if (magnetic_magnitude && *magnetic_magnitude) {
        for (int i = 0; i < nx; i++) {
            if ((*magnetic_magnitude)[i]) {
                free((*magnetic_magnitude)[i]);
            }
        }
        free(*magnetic_magnitude);
        *magnetic_magnitude = NULL;
    }
}

/**
 * Export data in a format compatible with the frontend
 * This function generates data structures that match the frontend expectations
 * including vector fields and magnitudes
 * 
 * @param sim Pointer to the simulation context
 * @param filename The output file name (JSON format)
 */
void mhd_export_frontend_data(MHDSimulation *sim, const char *filename) {
    if (!sim || !filename) return;
    
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    // Calculate vector magnitudes
    double **velocity_magnitude = NULL;
    double **magnetic_magnitude = NULL;
    mhd_calculate_vector_magnitudes(sim, &velocity_magnitude, &magnetic_magnitude);
    
    // Update all metrics
    mhd_update_all_metrics(sim);
    
    // Start JSON output
    fprintf(fp, "{\n");
    
    // Export simulation metrics
    fprintf(fp, "  \"metrics\": {\n");
    fprintf(fp, "    \"time\": %g,\n", sim->current_time);
    fprintf(fp, "    \"energy_kinetic\": %g,\n", sim->energy_kinetic);
    fprintf(fp, "    \"energy_magnetic\": %g,\n", sim->energy_magnetic);
    fprintf(fp, "    \"energy_thermal\": %g,\n", sim->energy_thermal);
    fprintf(fp, "    \"max_div_b\": %g\n", sim->max_div_b);
    fprintf(fp, "  },\n");
    
    // Export grid dimensions
    fprintf(fp, "  \"grid\": {\n");
    fprintf(fp, "    \"nx\": %d,\n", sim->grid_size_x);
    fprintf(fp, "    \"ny\": %d,\n", sim->grid_size_y);
    fprintf(fp, "    \"nz\": %d\n", sim->grid_size_z);
    fprintf(fp, "  },\n");
    
    // Export scalar fields
    fprintf(fp, "  \"density\": [\n");
    for (int i = 0; i < sim->grid_size_x; i++) {
        fprintf(fp, "    [");
        for (int j = 0; j < sim->grid_size_y; j++) {
            fprintf(fp, "%g%s", sim->grid[i][j][0].density, 
                    j < sim->grid_size_y - 1 ? ", " : "");
        }
        fprintf(fp, "]%s\n", i < sim->grid_size_x - 1 ? "," : "");
    }
    fprintf(fp, "  ],\n");
    
    fprintf(fp, "  \"pressure\": [\n");
    for (int i = 0; i < sim->grid_size_x; i++) {
        fprintf(fp, "    [");
        for (int j = 0; j < sim->grid_size_y; j++) {
            fprintf(fp, "%g%s", sim->grid[i][j][0].pressure, 
                    j < sim->grid_size_y - 1 ? ", " : "");
        }
        fprintf(fp, "]%s\n", i < sim->grid_size_x - 1 ? "," : "");
    }
    fprintf(fp, "  ],\n");
    
    fprintf(fp, "  \"temperature\": [\n");
    for (int i = 0; i < sim->grid_size_x; i++) {
        fprintf(fp, "    [");
        for (int j = 0; j < sim->grid_size_y; j++) {
            fprintf(fp, "%g%s", sim->grid[i][j][0].temperature, 
                    j < sim->grid_size_y - 1 ? ", " : "");
        }
        fprintf(fp, "]%s\n", i < sim->grid_size_x - 1 ? "," : "");
    }
    fprintf(fp, "  ],\n");
    
    // Export velocity field
    fprintf(fp, "  \"velocity\": [\n");
    for (int i = 0; i < sim->grid_size_x; i++) {
        fprintf(fp, "    [");
        for (int j = 0; j < sim->grid_size_y; j++) {
            // Store the magnitude of the velocity vector for scalar visualization
            double vx = sim->grid[i][j][0].velocity.x;
            double vy = sim->grid[i][j][0].velocity.y;
            double vz = sim->grid[i][j][0].velocity.z;
            double vmag = sqrt(vx*vx + vy*vy + vz*vz);
            fprintf(fp, "%g%s", vmag, j < sim->grid_size_y - 1 ? ", " : "");
        }
        fprintf(fp, "]%s\n", i < sim->grid_size_x - 1 ? "," : "");
    }
    fprintf(fp, "  ],\n");
    
    // Export velocity field as vector
    fprintf(fp, "  \"velocity_field\": [\n");
    for (int i = 0; i < sim->grid_size_x; i++) {
        fprintf(fp, "    [");
        for (int j = 0; j < sim->grid_size_y; j++) {
            fprintf(fp, "{\"vx\": %g, \"vy\": %g}%s", 
                    sim->grid[i][j][0].velocity.x, 
                    sim->grid[i][j][0].velocity.y,
                    j < sim->grid_size_y - 1 ? ", " : "");
        }
        fprintf(fp, "]%s\n", i < sim->grid_size_x - 1 ? "," : "");
    }
    fprintf(fp, "  ],\n");
    
    // Export magnetic field
    fprintf(fp, "  \"magnetic_field\": [\n");
    for (int i = 0; i < sim->grid_size_x; i++) {
        fprintf(fp, "    [");
        for (int j = 0; j < sim->grid_size_y; j++) {
            // Store the magnitude of the magnetic vector for scalar visualization
            double bx = sim->grid[i][j][0].magnetic.x;
            double by = sim->grid[i][j][0].magnetic.y;
            double bz = sim->grid[i][j][0].magnetic.z;
            double bmag = sqrt(bx*bx + by*by + bz*bz);
            fprintf(fp, "%g%s", bmag, j < sim->grid_size_y - 1 ? ", " : "");
        }
        fprintf(fp, "]%s\n", i < sim->grid_size_x - 1 ? "," : "");
    }
    fprintf(fp, "  ],\n");
    
    // Export magnetic field as vector
    fprintf(fp, "  \"magnetic_vector\": [\n");
    for (int i = 0; i < sim->grid_size_x; i++) {
        fprintf(fp, "    [");
        for (int j = 0; j < sim->grid_size_y; j++) {
            fprintf(fp, "{\"vx\": %g, \"vy\": %g}%s", 
                    sim->grid[i][j][0].magnetic.x, 
                    sim->grid[i][j][0].magnetic.y,
                    j < sim->grid_size_y - 1 ? ", " : "");
        }
        fprintf(fp, "]%s\n", i < sim->grid_size_x - 1 ? "," : "");
    }
    fprintf(fp, "  ],\n");
    
    // Export velocity magnitude
    fprintf(fp, "  \"velocity_magnitude\": [\n");
    for (int i = 0; i < sim->grid_size_x; i++) {
        fprintf(fp, "    [");
        for (int j = 0; j < sim->grid_size_y; j++) {
            fprintf(fp, "%g%s", velocity_magnitude[i][j], 
                    j < sim->grid_size_y - 1 ? ", " : "");
        }
        fprintf(fp, "]%s\n", i < sim->grid_size_x - 1 ? "," : "");
    }
    fprintf(fp, "  ],\n");
    
    // Export magnetic magnitude
    fprintf(fp, "  \"magnetic_magnitude\": [\n");
    for (int i = 0; i < sim->grid_size_x; i++) {
        fprintf(fp, "    [");
        for (int j = 0; j < sim->grid_size_y; j++) {
            fprintf(fp, "%g%s", magnetic_magnitude[i][j], 
                    j < sim->grid_size_y - 1 ? ", " : "");
        }
        fprintf(fp, "]%s\n", i < sim->grid_size_x - 1 ? "," : "");
    }
    fprintf(fp, "  ]\n");
    
    // End JSON
    fprintf(fp, "}\n");
    
    // Clean up
    mhd_free_vector_magnitudes(&velocity_magnitude, &magnetic_magnitude, sim->grid_size_x);
    
    fclose(fp);
    printf("Frontend-compatible data exported to: %s\n", filename);
}

void mhd_export_to_hdf5(MHDSimulation *sim, const char *filename) {
    if (!sim || !filename) return;
    

    printf("HDF5 export not implemented. Would export to: %s\n", filename);
    printf("To implement HDF5 export, add HDF5 library dependencies and implement this function.\n");
    
    
} 