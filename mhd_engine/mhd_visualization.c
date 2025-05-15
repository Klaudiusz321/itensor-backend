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


void mhd_export_to_hdf5(MHDSimulation *sim, const char *filename) {
    if (!sim || !filename) return;
    

    printf("HDF5 export not implemented. Would export to: %s\n", filename);
    printf("To implement HDF5 export, add HDF5 library dependencies and implement this function.\n");
    
    
} 