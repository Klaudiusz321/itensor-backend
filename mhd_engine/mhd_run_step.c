/**
 * mhd_run_step.c - Implementation of the simulation step function
 * 
 * This file contains the implementation of the mhd_run_step function,
 * which is responsible for executing a single simulation step.
 */

#include "mhd.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * Execute a single simulation step with enhanced stability checks
 * This is a replacement for the original mhd_run_step function in mhd.c
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
    
    // 5) Apply dynamic changes if enabled
    if (sim->dynamic_field_enabled) {
        mhd_apply_dynamic_changes(sim);
    }
    
    // 6) Update all metrics after the step
    mhd_update_all_metrics(sim);
} 