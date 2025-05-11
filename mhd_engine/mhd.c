/**
 * mhd.c - Main implementation of MHD simulation engine
 * 
 * Contains core functionality for simulation initialization and execution.
 */

#include "mhd.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* z mhd_utils.c */
extern void mhd_allocate_grids(MHDSimulation *sim);
extern void mhd_free_grids(MHDSimulation *sim);
extern void mhd_seed_rng(void);

/**
 * Initialize a new MHD simulation with the specified grid dimensions
 */
MHDSimulation* mhd_initialize(int grid_size_x, int grid_size_y, int grid_size_z) {
    if (grid_size_x > MAX_GRID_SIZE || grid_size_y > MAX_GRID_SIZE || grid_size_z > MAX_GRID_SIZE) {
        fprintf(stderr, "Error: Grid size exceeds maximum allowed size of %d\n", MAX_GRID_SIZE);
        return NULL;
    }

    // Jeśli ktoś próbuje zrobić 1-warstową siatkę w Z, wymuszamy 2
    if (grid_size_z < 2) {
        fprintf(stderr, "Warning: grid_size_z < 2, using grid_size_z = 2 internally.\n");
        grid_size_z = 2;
    }

    MHDSimulation *sim = malloc(sizeof(MHDSimulation));
    if (!sim) {
        fprintf(stderr, "Error: Failed to allocate memory for simulation\n");
        return NULL;
    }

    // Ustaw parametry
    sim->grid_size_x = grid_size_x;
    sim->grid_size_y = grid_size_y;
    sim->grid_size_z = grid_size_z;

    sim->time_step    = 0.01;
    sim->total_time   = 10.0;
    sim->current_time = 0.0;
    sim->method       = EULER;

    sim->initial_density     = 1.0;
    sim->initial_pressure    = 1.0;
    sim->initial_temperature = 1.0;
    sim->initial_velocity.x  = 0.0;
    sim->initial_velocity.y  = 0.0;
    sim->initial_velocity.z  = 0.0;

    sim->initial_magnetic_field.x = 0.0;
    sim->initial_magnetic_field.y = 0.0;
    sim->initial_magnetic_field.z = 0.0;
    sim->magnetic_conductivity    = 1.0;
    sim->magnetic_viscosity       = 0.1;

    sim->boundary_type = PERIODIC;
    sim->mode          = STATIC;

    sim->turbulence_intensity       = 0.0;
    sim->use_particle_interaction   = false;
    sim->use_spatial_gradients      = false;
    sim->random_disturbance_intensity = 0.0;

    sim->sound_speed = 1.0;

    // Alokacja siatek i seed RNG
    mhd_allocate_grids(sim);
    mhd_seed_rng();

    // Wypełnij wartościami początkowymi
    for (int i = 0; i < sim->grid_size_x; i++) {
        for (int j = 0; j < sim->grid_size_y; j++) {
            for (int k = 0; k < sim->grid_size_z; k++) {
                GridCell *cell = &sim->grid[i][j][k];
                cell->density     = sim->initial_density;
                cell->pressure    = sim->initial_pressure;
                cell->temperature = sim->initial_temperature;
                cell->velocity.x  = sim->initial_velocity.x;
                cell->velocity.y  = sim->initial_velocity.y;
                cell->velocity.z  = sim->initial_velocity.z;
                cell->magnetic.x  = sim->initial_magnetic_field.x;
                cell->magnetic.y  = sim->initial_magnetic_field.y;
                cell->magnetic.z  = sim->initial_magnetic_field.z;
            }
        }
    }

    return sim;
}

/**
 * Free all resources used by a simulation
 */
void mhd_free(MHDSimulation *sim) {
    if (!sim) return;
    // zwalniamy siatki
    mhd_free_grids(sim);
    free(sim);
}

/**
 * Execute a single simulation step
 */
void mhd_run_step(MHDSimulation *sim) {
    if (!sim) return;

    // 0) ewentualnie dynamiczny CFL tutaj, przed solverem:
    //    double max_speed = ...; sim->time_step = fmin(sim->time_step, 0.5*dx/max_speed);

    // 1) Boundary conditions na stan główny
    mhd_apply_boundary_conditions(sim, sim->grid);

    // 2) Wybór metody i wykonanie kroku
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
            mhd_solver_euler_step(sim);
    }

    // 3) Aktualizacja czasu symulacji
    sim->current_time += sim->time_step;
}

/**
 * main loop: wykonuj kroki aż do osiągnięcia total_time
 */
void mhd_run_simulation(MHDSimulation *sim) {
    if (!sim) return;

    while (sim->current_time < sim->total_time) {
        mhd_run_step(sim);
        // Możesz tu co jakiś czas wypisać status:
        mhd_print_status(sim);
    }
}

/**
 * Set the time step for the simulation
 */
void mhd_set_time_step(MHDSimulation *sim, double time_step) {
    if (!sim) return;
    if (time_step > 0.0) {
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
        sim->total_time = total_time;
    } else {
        fprintf(stderr, "Warning: Total time must be positive, using default value\n");
    }
}
