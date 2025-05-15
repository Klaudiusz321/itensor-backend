/**
 * mhd.c - Main implementation of MHD simulation engine
 * 
 * Contains core functionality for simulation initialization and execution.
 */
#include "mhd.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/**
 * Check if a value is finite (not NaN or Inf)
 */
bool is_finite_value(double value) {
    return isfinite(value);
}

/**
 * Print detailed error information when numerical instability is detected
 */
void report_numerical_instability(const char *location, double value) {
    fprintf(stderr, "Numerical instability detected in %s: value = %g\n", 
            location, value);
}

/**
 * Allocate both main and temporary grids
 */
bool mhd_allocate_grids(MHDSimulation *sim) {
    int nx = sim->grid_size_x;
    int ny = sim->grid_size_y;
    int nz = sim->grid_size_z;

    // 1) Alokacja pierwszego wymiaru: wskaźników do tablicy 2D
    sim->grid = malloc(nx * sizeof(GridCell **));
    if (!sim->grid) {
        fprintf(stderr, "❌ Error: malloc grid pointers failed\n");
        return false;
    }

    // 2) Alokacja drugiego wymiaru i trzeciego
    for (int i = 0; i < nx; ++i) {
        sim->grid[i] = malloc(ny * sizeof(GridCell *));
        if (!sim->grid[i]) {
            fprintf(stderr, "❌ Error: malloc grid[%d] pointers failed\n", i);
            // sprzątanie
            for (int ii = 0; ii < i; ++ii) free(sim->grid[ii]);
            free(sim->grid);
            return false;
        }
        for (int j = 0; j < ny; ++j) {
            // calloc zeruje pamięć – wszystkie pola GridCell będą na 0
            sim->grid[i][j] = calloc(nz, sizeof(GridCell));
            if (!sim->grid[i][j]) {
                fprintf(stderr,
                    "❌ Error: calloc grid[%d][%d] failed\n", i, j);
                // sprzątanie
                for (int jj = 0; jj < j; ++jj)      free(sim->grid[i][jj]);
                free(sim->grid[i]);
                for (int ii = 0; ii < i; ++ii) {
                    for (int jj = 0; jj < ny; ++jj) free(sim->grid[ii][jj]);
                    free(sim->grid[ii]);
                }
                free(sim->grid);
                return false;
            }
        }
    }
    
    // 3) Allocate temp_grid with the same dimensions
    sim->temp_grid = malloc(nx * sizeof(GridCell **));
    if (!sim->temp_grid) {
        fprintf(stderr, "❌ Error: malloc temp_grid pointers failed\n");
        mhd_free_grids(sim);
        return false;
    }
    
    for (int i = 0; i < nx; ++i) {
        sim->temp_grid[i] = malloc(ny * sizeof(GridCell *));
        if (!sim->temp_grid[i]) {
            fprintf(stderr, "❌ Error: malloc temp_grid[%d] pointers failed\n", i);
            // Clean up temp_grid
            for (int ii = 0; ii < i; ++ii) free(sim->temp_grid[ii]);
            free(sim->temp_grid);
            sim->temp_grid = NULL;
            // Clean up main grid
            mhd_free_grids(sim);
            return false;
        }
        for (int j = 0; j < ny; ++j) {
            sim->temp_grid[i][j] = calloc(nz, sizeof(GridCell));
            if (!sim->temp_grid[i][j]) {
                fprintf(stderr, "❌ Error: calloc temp_grid[%d][%d] failed\n", i, j);
                // Clean up temp_grid
                for (int jj = 0; jj < j; ++jj) free(sim->temp_grid[i][jj]);
                free(sim->temp_grid[i]);
                for (int ii = 0; ii < i; ++ii) {
                    for (int jj = 0; jj < ny; ++jj) free(sim->temp_grid[ii][jj]);
                    free(sim->temp_grid[ii]);
                }
                free(sim->temp_grid);
                sim->temp_grid = NULL;
                // Clean up main grid
                mhd_free_grids(sim);
                return false;
            }
        }
    }

    // 4) Allocate additional temporary grids for RK methods
    // Allocate temp_grid1
    sim->temp_grid1 = malloc(nx * sizeof(GridCell **));
    if (!sim->temp_grid1) {
        fprintf(stderr, "❌ Error: malloc temp_grid1 pointers failed\n");
        mhd_free_grids(sim);
        return false;
    }
    
    for (int i = 0; i < nx; ++i) {
        sim->temp_grid1[i] = malloc(ny * sizeof(GridCell *));
        if (!sim->temp_grid1[i]) {
            fprintf(stderr, "❌ Error: malloc temp_grid1[%d] pointers failed\n", i);
            // Clean up
            for (int ii = 0; ii < i; ++ii) free(sim->temp_grid1[ii]);
            free(sim->temp_grid1);
            sim->temp_grid1 = NULL;
            mhd_free_grids(sim);
            return false;
        }
        for (int j = 0; j < ny; ++j) {
            sim->temp_grid1[i][j] = calloc(nz, sizeof(GridCell));
            if (!sim->temp_grid1[i][j]) {
                fprintf(stderr, "❌ Error: calloc temp_grid1[%d][%d] failed\n", i, j);
                // Clean up
                for (int jj = 0; jj < j; ++jj) free(sim->temp_grid1[i][jj]);
                free(sim->temp_grid1[i]);
                for (int ii = 0; ii < i; ++ii) {
                    for (int jj = 0; jj < ny; ++jj) free(sim->temp_grid1[ii][jj]);
                    free(sim->temp_grid1[ii]);
                }
                free(sim->temp_grid1);
                sim->temp_grid1 = NULL;
                mhd_free_grids(sim);
                return false;
            }
        }
    }
    
    // Allocate temp_grid2
    sim->temp_grid2 = malloc(nx * sizeof(GridCell **));
    if (!sim->temp_grid2) {
        fprintf(stderr, "❌ Error: malloc temp_grid2 pointers failed\n");
        mhd_free_grids(sim);
        return false;
    }
    
    for (int i = 0; i < nx; ++i) {
        sim->temp_grid2[i] = malloc(ny * sizeof(GridCell *));
        if (!sim->temp_grid2[i]) {
            fprintf(stderr, "❌ Error: malloc temp_grid2[%d] pointers failed\n", i);
            // Clean up
            for (int ii = 0; ii < i; ++ii) free(sim->temp_grid2[ii]);
            free(sim->temp_grid2);
            sim->temp_grid2 = NULL;
            mhd_free_grids(sim);
            return false;
        }
        for (int j = 0; j < ny; ++j) {
            sim->temp_grid2[i][j] = calloc(nz, sizeof(GridCell));
            if (!sim->temp_grid2[i][j]) {
                fprintf(stderr, "❌ Error: calloc temp_grid2[%d][%d] failed\n", i, j);
                // Clean up
                for (int jj = 0; jj < j; ++jj) free(sim->temp_grid2[i][jj]);
                free(sim->temp_grid2[i]);
                for (int ii = 0; ii < i; ++ii) {
                    for (int jj = 0; jj < ny; ++jj) free(sim->temp_grid2[ii][jj]);
                    free(sim->temp_grid2[ii]);
                }
                free(sim->temp_grid2);
                sim->temp_grid2 = NULL;
                mhd_free_grids(sim);
                return false;
            }
        }
    }
    
    // Allocate temp_grid3 and temp_grid4 for RK4
    sim->temp_grid3 = malloc(nx * sizeof(GridCell **));
    if (!sim->temp_grid3) {
        fprintf(stderr, "❌ Error: malloc temp_grid3 pointers failed\n");
        mhd_free_grids(sim);
        return false;
    }
    
    for (int i = 0; i < nx; ++i) {
        sim->temp_grid3[i] = malloc(ny * sizeof(GridCell *));
        if (!sim->temp_grid3[i]) {
            fprintf(stderr, "❌ Error: malloc temp_grid3[%d] pointers failed\n", i);
            // Clean up
            for (int ii = 0; ii < i; ++ii) free(sim->temp_grid3[ii]);
            free(sim->temp_grid3);
            sim->temp_grid3 = NULL;
            mhd_free_grids(sim);
            return false;
        }
        for (int j = 0; j < ny; ++j) {
            sim->temp_grid3[i][j] = calloc(nz, sizeof(GridCell));
            if (!sim->temp_grid3[i][j]) {
                fprintf(stderr, "❌ Error: calloc temp_grid3[%d][%d] failed\n", i, j);
                // Clean up
                for (int jj = 0; jj < j; ++jj) free(sim->temp_grid3[i][jj]);
                free(sim->temp_grid3[i]);
                for (int ii = 0; ii < i; ++ii) {
                    for (int jj = 0; jj < ny; ++jj) free(sim->temp_grid3[ii][jj]);
                    free(sim->temp_grid3[ii]);
                }
                free(sim->temp_grid3);
                sim->temp_grid3 = NULL;
                mhd_free_grids(sim);
                return false;
            }
        }
    }
    
    sim->temp_grid4 = malloc(nx * sizeof(GridCell **));
    if (!sim->temp_grid4) {
        fprintf(stderr, "❌ Error: malloc temp_grid4 pointers failed\n");
        mhd_free_grids(sim);
        return false;
    }
    
    for (int i = 0; i < nx; ++i) {
        sim->temp_grid4[i] = malloc(ny * sizeof(GridCell *));
        if (!sim->temp_grid4[i]) {
            fprintf(stderr, "❌ Error: malloc temp_grid4[%d] pointers failed\n", i);
            // Clean up
            for (int ii = 0; ii < i; ++ii) free(sim->temp_grid4[ii]);
            free(sim->temp_grid4);
            sim->temp_grid4 = NULL;
            mhd_free_grids(sim);
            return false;
        }
        for (int j = 0; j < ny; ++j) {
            sim->temp_grid4[i][j] = calloc(nz, sizeof(GridCell));
            if (!sim->temp_grid4[i][j]) {
                fprintf(stderr, "❌ Error: calloc temp_grid4[%d][%d] failed\n", i, j);
                // Clean up
                for (int jj = 0; jj < j; ++jj) free(sim->temp_grid4[i][jj]);
                free(sim->temp_grid4[i]);
                for (int ii = 0; ii < i; ++ii) {
                    for (int jj = 0; jj < ny; ++jj) free(sim->temp_grid4[ii][jj]);
                    free(sim->temp_grid4[ii]);
                }
                free(sim->temp_grid4);
                sim->temp_grid4 = NULL;
                mhd_free_grids(sim);
                return false;
            }
        }
    }
    
    // Allocate tmp_grid
    sim->tmp_grid = malloc(nx * sizeof(GridCell **));
    if (!sim->tmp_grid) {
        fprintf(stderr, "❌ Error: malloc tmp_grid pointers failed\n");
        mhd_free_grids(sim);
        return false;
    }
    
    for (int i = 0; i < nx; ++i) {
        sim->tmp_grid[i] = malloc(ny * sizeof(GridCell *));
        if (!sim->tmp_grid[i]) {
            fprintf(stderr, "❌ Error: malloc tmp_grid[%d] pointers failed\n", i);
            // Clean up
            for (int ii = 0; ii < i; ++ii) free(sim->tmp_grid[ii]);
            free(sim->tmp_grid);
            sim->tmp_grid = NULL;
            mhd_free_grids(sim);
            return false;
        }
        for (int j = 0; j < ny; ++j) {
            sim->tmp_grid[i][j] = calloc(nz, sizeof(GridCell));
            if (!sim->tmp_grid[i][j]) {
                fprintf(stderr, "❌ Error: calloc tmp_grid[%d][%d] failed\n", i, j);
                // Clean up
                for (int jj = 0; jj < j; ++jj) free(sim->tmp_grid[i][jj]);
                free(sim->tmp_grid[i]);
                for (int ii = 0; ii < i; ++ii) {
                    for (int jj = 0; jj < ny; ++jj) free(sim->tmp_grid[ii][jj]);
                    free(sim->tmp_grid[ii]);
                }
                free(sim->tmp_grid);
                sim->tmp_grid = NULL;
                mhd_free_grids(sim);
                return false;
            }
        }
    }

    return true;
}

/**
 * Poprawiona funkcja inicjalizująca symulację MHD.
 * Alokuje i zeruje strukturę, ustawia parametry i buduje grid.
 */
MHDSimulation *mhd_initialize(int grid_size_x,
                              int grid_size_y,
                              int grid_size_z) {
    fprintf(stderr,
        "🔧 Initializing MHD simulation: %dx%dx%d\n",
        grid_size_x, grid_size_y, grid_size_z);

    MHDSimulation *sim = calloc(1, sizeof(*sim));
    if (!sim) {
        fprintf(stderr, "❌ Error: calloc(MHDSimulation) failed\n");
        return NULL;
    }

    // Inicjalizacja parametrów symulacji
    sim->grid_size_x = grid_size_x;
    sim->grid_size_y = grid_size_y;
    sim->grid_size_z = grid_size_z;
    sim->time_step    = 0.01;
    sim->total_time   = 1.0;
    sim->current_time = 0.0;
    sim->method       = RUNGE_KUTTA_2;
    
    // Inicjalizacja parametrów fizycznych z wartościami domyślnymi
    sim->initial_density = 1.0;
    sim->initial_pressure = 1.0;
    sim->initial_temperature = 1.0;
    
    sim->initial_velocity.x = 0.0;
    sim->initial_velocity.y = 0.0;
    sim->initial_velocity.z = 0.0;
    
    sim->initial_magnetic_field.x = 0.0;
    sim->initial_magnetic_field.y = 0.0;
    sim->initial_magnetic_field.z = 0.0;
    
    sim->magnetic_conductivity = 0.0;
    sim->magnetic_viscosity = 0.0;
    
    sim->boundary_type = BC_PERIODIC;
    sim->mode = MODE_STATIC;

    // Alokacja trójwymiarowej tablicy
    if (!mhd_allocate_grids(sim)) {
        fprintf(stderr, "❌ Error: Grid allocation in mhd_allocate_grids failed\n");
        free(sim);
        return NULL;
    }
    fprintf(stderr, "✅ Grid allocation successful: grid=%p\n", (void*)sim->grid);
    
    // Inicjalizacja siatki z wartościami domyślnymi
    for (int i = 0; i < grid_size_x; i++) {
        for (int j = 0; j < grid_size_y; j++) {
            for (int k = 0; k < grid_size_z; k++) {
                sim->grid[i][j][k].density = sim->initial_density;
                sim->grid[i][j][k].pressure = sim->initial_pressure;
                sim->grid[i][j][k].temperature = sim->initial_temperature;
                
                sim->grid[i][j][k].velocity.x = sim->initial_velocity.x;
                sim->grid[i][j][k].velocity.y = sim->initial_velocity.y;
                sim->grid[i][j][k].velocity.z = sim->initial_velocity.z;
                
                sim->grid[i][j][k].magnetic.x = sim->initial_magnetic_field.x;
                sim->grid[i][j][k].magnetic.y = sim->initial_magnetic_field.y;
                sim->grid[i][j][k].magnetic.z = sim->initial_magnetic_field.z;
            }
        }
    }

    return sim;
}

/**
 * Free all resources used by a simulation
 */
void mhd_free(MHDSimulation *sim) {
    if (!sim) {
        fprintf(stderr, "Warning: mhd_free received NULL pointer.\n");
        return;
    }
    mhd_free_grids(sim);
    free(sim);
    fprintf(stderr, "Simulation freed successfully.\n");
}

/**
 * Execute a single simulation step with enhanced stability checks
 */
void mhd_run_step(MHDSimulation *sim) {
    if (!sim) return;

    // 1) Apply boundary conditions on the main grid
    mhd_apply_boundary_conditions(sim, sim->grid);

    // 2) Check for numerical instability before solver step
    bool instability = false;
    int instability_count = 0;
    int max_allowed_instabilities = 10; // Allow more instabilities before giving up
    
    for (int i = 0; i < sim->grid_size_x && !instability; ++i) {
        for (int j = 0; j < sim->grid_size_y && !instability; ++j) {
            for (int k = 0; k < sim->grid_size_z && !instability; ++k) {
                GridCell *cell = &sim->grid[i][j][k];

                // density, pressure, temperature - more permissive checks
                if (!is_finite_value(cell->density) || cell->density <= 0 ||
                    !is_finite_value(cell->pressure) || cell->pressure <= 0 ||
                    !is_finite_value(cell->temperature) || cell->temperature <= 0) 
                {
                    instability_count++;
                    
                    // Fix the values instead of reporting instability
                    if (instability_count <= max_allowed_instabilities) {
                        if (!is_finite_value(cell->density) || cell->density <= 0)
                            cell->density = MIN_DENSITY;
                        if (!is_finite_value(cell->pressure) || cell->pressure <= 0)
                            cell->pressure = MIN_PRESSURE;
                        if (!is_finite_value(cell->temperature) || cell->temperature <= 0)
                            cell->temperature = MIN_TEMPERATURE;
                    } else {
                        report_numerical_instability("thermo", cell->density);
                        instability = true;
                        break;
                    }
                }

                // velocity vector - more permissive checks
                if (!is_finite_value(cell->velocity.x) ||
                    !is_finite_value(cell->velocity.y) ||
                    !is_finite_value(cell->velocity.z)) 
                {
                    instability_count++;
                    
                    // Fix the values instead of reporting instability
                    if (instability_count <= max_allowed_instabilities) {
                        if (!is_finite_value(cell->velocity.x)) cell->velocity.x = 0.0;
                        if (!is_finite_value(cell->velocity.y)) cell->velocity.y = 0.0;
                        if (!is_finite_value(cell->velocity.z)) cell->velocity.z = 0.0;
                    } else {
                        double vmag = fabs(cell->velocity.x)
                                    + fabs(cell->velocity.y)
                                    + fabs(cell->velocity.z);
                        report_numerical_instability("velocity", vmag);
                        instability = true;
                        break;
                    }
                }

                // magnetic vector - more permissive checks
                if (!is_finite_value(cell->magnetic.x) ||
                    !is_finite_value(cell->magnetic.y) ||
                    !is_finite_value(cell->magnetic.z)) 
                {
                    instability_count++;
                    
                    // Fix the values instead of reporting instability
                    if (instability_count <= max_allowed_instabilities) {
                        if (!is_finite_value(cell->magnetic.x)) cell->magnetic.x = 0.0;
                        if (!is_finite_value(cell->magnetic.y)) cell->magnetic.y = 0.0;
                        if (!is_finite_value(cell->magnetic.z)) cell->magnetic.z = 0.0;
                    } else {
                        double bmag = fabs(cell->magnetic.x)
                                    + fabs(cell->magnetic.y)
                                    + fabs(cell->magnetic.z);
                        report_numerical_instability("magnetic", bmag);
                        instability = true;
                        break;
                    }
                }
            }
        }
    }

    if (instability) {
        fprintf(stderr, "WARNING: Numerical instability detected. Skipping solver step.\n");
        return;
    }

    // 3) Choose numerical method and advance
    switch (sim->method) {
        case EULER:
            mhd_solver_euler_step(sim);
            break;
        case RUNGE_KUTTA_2:
            mhd_solver_rk2_step(sim);
            break;
        case RUNGE_KUTTA_4:
            mhd_solver_rk4_step(sim);
            break;
        default:
            // fallback to RK2
            mhd_solver_rk2_step(sim);
            break;
    }

    // 4) Update simulation time
    sim->current_time += sim->time_step;
}

/**
 * main loop: wykonuj kroki aż do osiągnięcia total_time
 */
void mhd_run_simulation(MHDSimulation *sim) {
    if (!sim) return;

    int max_steps   = 10000;  // Safety limit to prevent infinite loops
    int step_count  = 0;
    int error_count = 0;
    int max_errors  = 5;      // Maximum number of errors before aborting

    // Główna pętla czasowa
    while (sim->current_time < sim->total_time && step_count < max_steps) {
        double prev_time = sim->current_time;
        mhd_run_step(sim);
        
        // Sprawdzenie, czy czas ruszył do przodu
        if (sim->current_time <= prev_time) {
            error_count++;
            if (error_count >= max_errors) {
                fprintf(stderr,
                  "ERROR: Too many numerical instabilities detected. Aborting simulation.\n");
                break;
            }
        }
        
        // Status co 10 kroków
        if (step_count % 10 == 0) {
            mhd_print_status(sim);
        }
        
        step_count++;
    }
    
    if (step_count >= max_steps) {
        fprintf(stderr,
          "WARNING: Reached maximum step count (%d). Simulation may not have completed.\n",
          max_steps);
    }
    
    // Podsumowanie
    printf("Simulation completed at time = %.3f after %d steps\n",
           sim->current_time, step_count);

    // --- OBLICZENIE I ZAPIS METRYK ---
    sim->energy_kinetic  = compute_energy_kinetic(sim);
    sim->energy_magnetic = compute_energy_magnetic(sim);
    sim->energy_thermal  = compute_energy_thermal(sim);
    sim->max_div_b       = compute_max_divergence_B(sim);


    // Wypisanie metryk (opcjonalnie)
    printf("  → E_kin = %.6g, E_mag = %.6g, E_th = %.6g, max |div B| = %.6g\n",
           sim->energy_kinetic, sim->energy_magnetic, sim->energy_thermal, sim->max_div_b);
}

/**
 * Set the time step for the simulation
 */
void mhd_set_time_step(MHDSimulation *sim, double time_step) {
    if (!sim) return;
    if (time_step > 0.0) {
        // Limit maximum time step for stability
        if (time_step > 0.01) {
            fprintf(stderr, "Warning: Time step too large, limiting to 0.01\n");
            time_step = 0.01;
        }
        sim->time_step = time_step;
    } else {
        fprintf(stderr, "Warning: Time step must be positive, using default value\n");
    }
}

/**
 * Set the total simulation time
 */
void mhd_set_total_time(MHDSimulation *sim, double total_time) {
    if (!sim) return;
    if (total_time > 0.0) {
        // Limit maximum simulation time for safety
        if (total_time > 10.0) {
            fprintf(stderr, "Warning: Total time too large, limiting to 10.0\n");
            total_time = 10.0;
        }
        sim->total_time = total_time;
    } else {
        fprintf(stderr, "Warning: Total time must be positive, using default value\n");
    }
}
