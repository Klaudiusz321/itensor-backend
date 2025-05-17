/*
 * mhd.h - Header file for MHD simulation engine
 *
 * Contains structures and function declarations for MHD simulation.
 */

#ifndef MHD_H
#define MHD_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

/* Constants */
#define MAX_GRID_SIZE       500
#define MAX_FILENAME_LENGTH 256
#define PI                  3.14159265358979323846

/* Stability thresholds */
#define MIN_DENSITY      1e-6
#define MAX_DENSITY      1e6
#define MIN_PRESSURE     1e-6
#define MAX_PRESSURE     1e6
#define MIN_TEMPERATURE  1e-6
#define MAX_TEMPERATURE  1e6
#define MAX_VELOCITY     1e3
#define MAX_MAGNETIC     1e3
#define STABILITY_FACTOR 0.8

/* Helper macros */
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

/* Enumerations */
typedef enum {
    EULER = 0,
    RUNGE_KUTTA_2 = 1,
    RUNGE_KUTTA_4 = 2
} NumericalMethod;

typedef enum {
    BC_OPEN = 0,
    BC_CLOSED = 1,
    BC_REFLECTIVE = 2,
    BC_PERIODIC = 3,
    BC_CUSTOM = 4
} BoundaryType;

typedef enum {
    MODE_STATIC,
    MODE_DYNAMIC,
    MODE_DISTURBED
} SimulationMode;

/* 3-vector */
typedef struct {
    double x, y, z;
} Vector3D;

/* One grid cell */
typedef struct {
    double   density;
    double   pressure;
    double   temperature;
    Vector3D velocity;
    Vector3D magnetic;
} GridCell;

/* Forward declare simulation struct for callbacks */
typedef struct MHDSimulation MHDSimulation;

/* Custom boundary callback */
typedef void (*CustomBoundaryFunc)(MHDSimulation *sim, GridCell ***buf);

/* Main simulation context */
struct MHDSimulation {
    /* Grid dimensions */
    int grid_size_x, grid_size_y, grid_size_z;

    /* Time stepping */
    double time_step;
    double total_time;
    double current_time;

    /* Energy and diagnostics */
    double sound_speed;
    double energy_kinetic;
    double energy_magnetic;
    double energy_thermal;
    double max_div_b;

    /* Numerical method */
    NumericalMethod method;

    /* Fluid initialization */
    double    initial_density;
    double    initial_pressure;
    double    initial_temperature;
    Vector3D  initial_velocity;

    /* Magnetic initialization */
    Vector3D  initial_magnetic_field;
    double    magnetic_conductivity;
    double    magnetic_viscosity;

    /* Boundary conditions */
    BoundaryType       boundary_type;
    CustomBoundaryFunc custom_boundary_func;

    /* Simulation mode & extras */
    SimulationMode mode;
    double         turbulence_intensity;
    bool           turbulence_enabled;
    bool           use_particle_interaction;
    bool           use_spatial_gradients;
    bool           disturbance_enabled;
    double         random_disturbance_intensity;
    double         disturbance_frequency;
    double         disturbance_magnitude;
    bool           random_disturbances_enabled;
    double         random_disturbance_magnitude;

    /* Dynamic-field parameters */
    bool   dynamic_field_enabled;
    int    dynamic_field_type;
    double dynamic_field_amplitude;
    double dynamic_field_frequency;

    /* Grids and temporaries */
    GridCell ***grid;
    GridCell ***temp_grid;
    GridCell ***temp_grid1;
    GridCell ***temp_grid2;
    GridCell ***temp_grid3;
    GridCell ***temp_grid4;
    GridCell ***tmp_grid;

    /* Gradient flags */
    bool gradient_density_enabled;
    bool gradient_velocity_enabled;
    bool gradient_magnetic_enabled;
    bool gradient_temperature_enabled;
    bool gradient_pressure_enabled;
};

/* Utility functions */


/* Initialization and cleanup */
MHDSimulation* mhd_initialize(int nx, int ny, int nz);
void           mhd_free(MHDSimulation *sim);

/* Time control */
void mhd_set_time_step(MHDSimulation *sim, double dt);
void mhd_set_total_time(MHDSimulation *sim, double total);

/* Grid allocation */
bool mhd_allocate_grids(MHDSimulation *sim);
void mhd_free_grids(MHDSimulation *sim);

/* Solvers */
void mhd_set_numerical_method(MHDSimulation *sim, NumericalMethod m);
void mhd_solver_euler_step(MHDSimulation *sim);
void mhd_solver_rk2_step(MHDSimulation *sim);
void mhd_solver_rk4_step(MHDSimulation *sim);
void mhd_update_derivatives(MHDSimulation *sim, GridCell ***source, GridCell ***derivatives);

/* Boundary conditions */
void mhd_apply_boundary_conditions(MHDSimulation *sim, GridCell ***buf);
void mhd_set_open_boundaries(MHDSimulation *sim, GridCell ***buf);
void mhd_set_closed_boundaries(MHDSimulation *sim, GridCell ***buf);
void mhd_set_reflective_boundaries(MHDSimulation *sim, GridCell ***buf);
void mhd_set_periodic_boundaries(MHDSimulation *sim, GridCell ***buf);
void mhd_set_custom_boundaries(MHDSimulation *sim, CustomBoundaryFunc f);

/* Main loop */
void mhd_run_step(MHDSimulation *sim);
void mhd_run_simulation(MHDSimulation *sim);

/* Fluid utilities */
void mhd_initialize_fluid_parameters(MHDSimulation *sim, double density, double pressure, double temperature);
void mhd_set_initial_velocity(MHDSimulation *sim, double vx, double vy, double vz);

/* Magnetic utilities */
void mhd_initialize_magnetic_field(MHDSimulation *sim, double bx, double by, double bz);
void mhd_update_magnetic_field(MHDSimulation *sim);
void mhd_apply_magnetic_viscosity(MHDSimulation *sim);
void mhd_apply_magnetic_conductivity(MHDSimulation *sim);
void mhd_set_magnetic_conductivity(MHDSimulation *sim, double cond);
void mhd_set_magnetic_viscosity(MHDSimulation *sim, double vis);

/* Utilities for export and diagnostics */
void   mhd_export_to_binary(MHDSimulation *sim, const char *fn);
void   mhd_export_to_hdf5(MHDSimulation *sim, const char *fn);
void   mhd_export_slice_data(MHDSimulation *sim, int slice_dim, int slice_pos, const char *fn);
void   mhd_export_field_data(MHDSimulation *sim, const char *field_name, const char *fn);
void   mhd_export_frontend_data(MHDSimulation *sim, const char *fn);
void   mhd_calculate_vector_magnitudes(MHDSimulation *sim, double ***velocity_magnitude, double ***magnetic_magnitude);
void   mhd_free_vector_magnitudes(double ***velocity_magnitude, double ***magnetic_magnitude, int nx);

double mhd_get_simulation_time(MHDSimulation *sim);
double mhd_get_energy_kinetic(MHDSimulation *sim);
double mhd_get_energy_magnetic(MHDSimulation *sim);
double mhd_get_energy_thermal(MHDSimulation *sim);
double mhd_get_max_div_b(MHDSimulation *sim);
void   mhd_update_all_metrics(MHDSimulation *sim);
void   mhd_apply_dynamic_changes(MHDSimulation *sim);
/* Funkcje obliczające faktyczne wartości energii i dywergencji */
double compute_energy_kinetic(MHDSimulation *sim);
double compute_energy_magnetic(MHDSimulation *sim);
double compute_energy_thermal(MHDSimulation *sim);
double compute_max_divergence_B(MHDSimulation *sim);

void mhd_print_status(MHDSimulation *sim);
void mhd_add_random_disturbance(MHDSimulation *sim, double intensity);
double mhd_random_value(double min, double max);
void mhd_apply_turbulence(MHDSimulation *sim, double intensity);
void mhd_apply_spatial_gradients(MHDSimulation *sim, bool enable);

/* Funkcje pomocnicze do obsługi stabilności numerycznej */
bool is_finite_value(double value);
void report_numerical_instability(const char *location, double value);

static inline void enforce_physical_constraints(GridCell *cell) {
    if (!cell) return;
    
    // Density
    if (!isfinite(cell->density) || cell->density < MIN_DENSITY)
        cell->density = MIN_DENSITY;
    else if (cell->density > MAX_DENSITY)
        cell->density = MAX_DENSITY;
    
    // Pressure
    if (!isfinite(cell->pressure) || cell->pressure < MIN_PRESSURE)
        cell->pressure = MIN_PRESSURE;
    else if (cell->pressure > MAX_PRESSURE)
        cell->pressure = MAX_PRESSURE;
    
    // Temperature
    if (!isfinite(cell->temperature) || cell->temperature < MIN_TEMPERATURE)
        cell->temperature = MIN_TEMPERATURE;
    else if (cell->temperature > MAX_TEMPERATURE)
        cell->temperature = MAX_TEMPERATURE;
    
    // Velocity components
    cell->velocity.x = MAX(MIN(isfinite(cell->velocity.x) ? cell->velocity.x : 0.0, MAX_VELOCITY), -MAX_VELOCITY);
    cell->velocity.y = MAX(MIN(isfinite(cell->velocity.y) ? cell->velocity.y : 0.0, MAX_VELOCITY), -MAX_VELOCITY);
    cell->velocity.z = MAX(MIN(isfinite(cell->velocity.z) ? cell->velocity.z : 0.0, MAX_VELOCITY), -MAX_VELOCITY);
    
    // Magnetic components
    cell->magnetic.x = MAX(MIN(isfinite(cell->magnetic.x) ? cell->magnetic.x : 0.0, MAX_MAGNETIC), -MAX_MAGNETIC);
    cell->magnetic.y = MAX(MIN(isfinite(cell->magnetic.y) ? cell->magnetic.y : 0.0, MAX_MAGNETIC), -MAX_MAGNETIC);
    cell->magnetic.z = MAX(MIN(isfinite(cell->magnetic.z) ? cell->magnetic.z : 0.0, MAX_MAGNETIC), -MAX_MAGNETIC);
}

static inline void apply_divergence_cleaning(GridCell *d, double div_B, double dt) {
    if (!d || !isfinite(div_B) || !isfinite(dt) || dt <= 0) return;

    double cleaning_factor = 0.1;
    double corr = -cleaning_factor * div_B * dt;
    double max_corr = 0.01;
    corr = (corr > max_corr) ? max_corr : (corr < -max_corr ? -max_corr : corr);

    d->magnetic.x -= corr / 3.0;
    d->magnetic.y -= corr / 3.0;
    d->magnetic.z -= corr / 3.0;
}

#endif /* MHD_H */