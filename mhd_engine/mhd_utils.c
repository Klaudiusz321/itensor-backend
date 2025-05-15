/**
 * mhd_utils.c – Utility functions for MHD simulation
 *
 * Contains helper functions for randomness, status printing, and RNG seeding.
 */

#include "mhd.h"
#include <stdlib.h>   // calloc, free, rand, srand
#include <stdio.h>    // fprintf, stderr
#include <time.h>     // time
#include <math.h>     // isfinite

// Ensure RNG is seeded only once
static int mhd_rng_seeded = 0;

/**
 * Seed RNG once per simulation.
 * Call this at the start of mhd_initialize().
 */
void mhd_seed_rng(void) {
    if (!mhd_rng_seeded) {
        srand((unsigned int)time(NULL));
        mhd_rng_seeded = 1;
    }
}

/**
 * Generate a random value in [min, max] using rand().
 */
double mhd_random_value(double min, double max) {
    double t = (double)rand() / (double)RAND_MAX;
    return min + t * (max - min);
}

/**
 * Add a random disturbance of given intensity to density and velocity fields.
 */
void mhd_add_random_disturbance(MHDSimulation *sim, double intensity) {
    if (!sim || intensity <= 0.0) return;
    int nx = sim->grid_size_x;
    int ny = sim->grid_size_y;
    int nz = sim->grid_size_z;

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                GridCell *cell = &sim->grid[i][j][k];

                // Apply random density disturbance
                double d = mhd_random_value(-intensity, intensity);
                cell->density += d;

                // Apply random velocity disturbance
                cell->velocity.x += mhd_random_value(-intensity, intensity);
                cell->velocity.y += mhd_random_value(-intensity, intensity);
                cell->velocity.z += mhd_random_value(-intensity, intensity);

                // Clamp density to finite non-negative
                if (!isfinite(cell->density) || cell->density < 0.0) {
                    cell->density = 0.0;
                }
            }
        }
    }
}

/**
 * Print current simulation status (time, method, mode, boundary) to stderr.
 */
void mhd_print_status(MHDSimulation *sim) {
    if (!sim) return;
    fprintf(stderr,
        "[MHD] time: %g / %g, method: %d, mode: %d, boundary: %d\n",
        sim->current_time,
        sim->total_time,
        (int)sim->method,
        (int)sim->mode,
        (int)sim->boundary_type
    );
}

void mhd_free_grids(MHDSimulation *sim) {
    if (!sim) return;
    int nx = sim->grid_size_x;
    int ny = sim->grid_size_y;
    
    // Funkcja zwalniająca siatkę 3D
    void free_grid_3d(GridCell ***grid) {
        if (!grid) return;
        for (int i = 0; i < nx; ++i) {
            if (grid[i]) {
                for (int j = 0; j < ny; ++j) {
                    free(grid[i][j]);
                }
                free(grid[i]);
            }
        }
        free(grid);
    }
    
    // Zwolnij główną siatkę
    free_grid_3d(sim->grid);
    sim->grid = NULL;
    
    // Zwolnij wszystkie siatki tymczasowe
    free_grid_3d(sim->temp_grid);
    sim->temp_grid = NULL;
    
    free_grid_3d(sim->temp_grid1);
    sim->temp_grid1 = NULL;
    
    free_grid_3d(sim->temp_grid2);
    sim->temp_grid2 = NULL;
    
    free_grid_3d(sim->temp_grid3);
    sim->temp_grid3 = NULL;
    
    free_grid_3d(sim->temp_grid4);
    sim->temp_grid4 = NULL;
    
    free_grid_3d(sim->tmp_grid);
    sim->tmp_grid = NULL;
}

double compute_energy_kinetic(MHDSimulation *sim) {
    double E = 0.0;
    for (int i = 0; i < sim->grid_size_x; ++i) {
      for (int j = 0; j < sim->grid_size_y; ++j) {
        for (int k = 0; k < sim->grid_size_z; ++k) {
          GridCell *c = &sim->grid[i][j][k];
          double vx = c->velocity.x,
                 vy = c->velocity.y,
                 vz = c->velocity.z;
          double v2 = vx*vx + vy*vy + vz*vz;
          E += 0.5 * c->density * v2;
        }
      }
    }
    return E;
}

/** Oblicza energię magnetyczną: ½·|B|² (zakładamy μ₀=1) */
double compute_energy_magnetic(MHDSimulation *sim) {
    double E = 0.0;
    for (int i = 0; i < sim->grid_size_x; ++i) {
      for (int j = 0; j < sim->grid_size_y; ++j) {
        for (int k = 0; k < sim->grid_size_z; ++k) {
          GridCell *c = &sim->grid[i][j][k];
          double bx = c->magnetic.x,
                 by = c->magnetic.y,
                 bz = c->magnetic.z;
          double b2 = bx*bx + by*by + bz*bz;
          E += 0.5 * b2;
        }
      }
    }
    return E;
}

/** Oblicza energię termiczną: ∑ p/(γ−1) zakładając γ=5/3 */
double compute_energy_thermal(MHDSimulation *sim) {
    const double gamma = 5.0/3.0;
    double E = 0.0;
    for (int i = 0; i < sim->grid_size_x; ++i) {
      for (int j = 0; j < sim->grid_size_y; ++j) {
        for (int k = 0; k < sim->grid_size_z; ++k) {
          GridCell *c = &sim->grid[i][j][k];
          E += c->pressure / (gamma - 1.0);
        }
      }
    }
    return E;
}

double divB(MHDSimulation *sim, int i, int j, int k) {
    double dx = 1.0, dy = 1.0, dz = 1.0;
    int nx = sim->grid_size_x, ny = sim->grid_size_y, nz = sim->grid_size_z;
    
    // granice – proste warunki okresowe lub zerowe
    int im = (i-1+nx)%nx, ip = (i+1)%nx;
    int jm = (j-1+ny)%ny, jp = (j+1)%ny;
    int km = (k-1+nz)%nz, kp = (k+1)%nz;
    
    GridCell
      *c_im = &sim->grid[im][j][k],
      *c_ip = &sim->grid[ip][j][k],
      *c_jm = &sim->grid[i][jm][k],
      *c_jp = &sim->grid[i][jp][k],
      *c_km = &sim->grid[i][j][km],
      *c_kp = &sim->grid[i][j][kp];
    
    double dBx_dx = (c_ip->magnetic.x - c_im->magnetic.x) / (2*dx);
    double dBy_dy = (c_jp->magnetic.y - c_jm->magnetic.y) / (2*dy);
    double dBz_dz = (c_kp->magnetic.z - c_km->magnetic.z) / (2*dz);
    
    return dBx_dx + dBy_dy + dBz_dz;
}

/** Prosty pomiar max |∇·B|, zakładamy dostęp do funkcji divB(sim,i,j,k) */
double compute_max_divergence_B(MHDSimulation *sim) {
    double max_div = 0.0;
    for (int i = 0; i < sim->grid_size_x; ++i) {
      for (int j = 0; j < sim->grid_size_y; ++j) {
        for (int k = 0; k < sim->grid_size_z; ++k) {
          double d = fabs(divB(sim, i, j, k));
          if (d > max_div) max_div = d;
        }
      }
    }
    return max_div;
}

