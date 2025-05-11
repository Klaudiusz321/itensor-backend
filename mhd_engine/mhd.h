/**
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
#define MAX_GRID_SIZE 500
#define MAX_FILENAME_LENGTH 256
#define PI 3.14159265358979323846

/* Enumeration types */
typedef enum {
    EULER,
    RUNGE_KUTTA_2,
    RUNGE_KUTTA_4
} NumericalMethod;

typedef enum {
    OPEN,
    CLOSED,
    REFLECTIVE,
    PERIODIC,
    CUSTOM
} BoundaryType;

typedef enum {
    STATIC,
    DYNAMIC,
    DISTURBED
} SimulationMode;

/* Vector structure */
typedef struct {
    double x;
    double y;
    double z;
} Vector3D;

/* Grid cell structure */
typedef struct {
    double density;      /* Fluid density */
    double pressure;     /* Fluid pressure */
    double temperature;  /* Fluid temperature */
    Vector3D velocity;   /* Fluid velocity */
    Vector3D magnetic;   /* Magnetic field */
} GridCell;

/* Forward declaration of MHDSimulation for callback typedef */
struct MHDSimulation;

/* Custom boundary function type */
typedef void (*CustomBoundaryFunc)(struct MHDSimulation *);

/* Simulation parameters structure */
typedef struct MHDSimulation {
    /* General simulation parameters */
    int grid_size_x;
    int grid_size_y;
    int grid_size_z;
    double time_step;
    double total_time;
    double current_time;
    NumericalMethod method;

    double sound_speed;
    
    /* Fluid parameters */
    double initial_density;
    double initial_pressure;
    double initial_temperature;
    Vector3D initial_velocity;
    
    /* Magnetic field parameters */
    Vector3D initial_magnetic_field;
    double magnetic_conductivity;
    double magnetic_viscosity;
    
    /* Boundary conditions */
    BoundaryType boundary_type;
    CustomBoundaryFunc custom_boundary_func;  /* <-- added for CUSTOM mode */
    
    /* Simulation mode */
    SimulationMode mode;
    
    /* Advanced options */
    double turbulence_intensity;
    bool use_particle_interaction;
    bool use_spatial_gradients;
    double random_disturbance_intensity;
    
    /* Grid data */
    GridCell ***grid;
    GridCell ***temp_grid;
    GridCell ***temp_grid1;
    GridCell ***temp_grid2;
    GridCell ***temp_grid3;
    GridCell ***temp_grid4;
    GridCell ***tmp_grid;
     /* Used for intermediate steps in numerical methods */
    /* If you implement RK2/RK4 buffers, add pointers for k1,k2,k3,k4 here */
} MHDSimulation;

/* Function declarations */
/* mhd.c - Main simulation functions */
MHDSimulation*  mhd_initialize(int grid_size_x, int grid_size_y, int grid_size_z);
void            mhd_free(MHDSimulation *sim);
void            mhd_run_step(MHDSimulation *sim);
void            mhd_run_simulation(MHDSimulation *sim);
void            mhd_set_time_step(MHDSimulation *sim, double time_step);
void            mhd_set_total_time(MHDSimulation *sim, double total_time);

/* mhd_solver.c - Numerical methods */
void            mhd_set_numerical_method(MHDSimulation *sim, NumericalMethod method);
void            mhd_solver_euler_step(MHDSimulation *sim);
void            mhd_solver_rk2_step(MHDSimulation *sim);
void            mhd_solver_rk4_step(MHDSimulation *sim);
void            mhd_update_derivatives(MHDSimulation *sim, GridCell ***source, GridCell ***derivatives);

/* mhd_conditions.c - Boundary conditions */
void            mhd_set_boundary_type(MHDSimulation *sim, BoundaryType type);
void            mhd_set_custom_boundaries(MHDSimulation *sim, CustomBoundaryFunc func);
void            mhd_apply_boundary_conditions(MHDSimulation *sim, GridCell ***buf);
void            mhd_set_open_boundaries(MHDSimulation *sim);
void            mhd_set_closed_boundaries(MHDSimulation *sim);
void            mhd_set_reflective_boundaries(MHDSimulation *sim);
void            mhd_set_periodic_boundaries(MHDSimulation *sim);

/* mhd_magnetic.c - Magnetic field functions */
void            mhd_initialize_magnetic_field(MHDSimulation *sim, double bx, double by, double bz);
void            mhd_update_magnetic_field(MHDSimulation *sim);
void            mhd_apply_magnetic_viscosity(MHDSimulation *sim);
void            mhd_apply_magnetic_conductivity(MHDSimulation *sim);
void            mhd_set_magnetic_conductivity(MHDSimulation *sim, double conductivity);
void            mhd_set_magnetic_viscosity(MHDSimulation *sim, double viscosity);

/* mhd_fluid.c - Fluid dynamics functions */
void            mhd_initialize_fluid_parameters(MHDSimulation *sim, double density, double pressure, double temperature);
void            mhd_set_initial_velocity(MHDSimulation *sim, double vx, double vy, double vz);
void            mhd_update_fluid_dynamics(MHDSimulation *sim);
void            mhd_apply_temperature_gradient(MHDSimulation *sim, double gradient_x, double gradient_y, double gradient_z);
void            mhd_apply_pressure_gradient(MHDSimulation *sim, double gradient_x, double gradient_y, double gradient_z);

/* mhd_advanced.c - Advanced simulation options */
void            mhd_set_simulation_mode(MHDSimulation *sim, SimulationMode mode);
void            mhd_apply_disturbances(MHDSimulation *sim, double intensity);
void            mhd_apply_dynamic_changes(MHDSimulation *sim);
void            mhd_apply_turbulence(MHDSimulation *sim, double intensity);
void            mhd_apply_particle_interaction(MHDSimulation *sim, bool enable);
void            mhd_apply_spatial_gradients(MHDSimulation *sim, bool enable);
void            mhd_set_random_disturbance_intensity(MHDSimulation *sim, double intensity);

/* mhd_utils.c - Utility functions */
void            mhd_allocate_grid(MHDSimulation *sim);
void            mhd_allocate_temp_grid(MHDSimulation *sim);
void            mhd_copy_grid(MHDSimulation *sim, GridCell ***source, GridCell ***destination);
double          mhd_random_value(double min, double max);
void            mhd_add_random_disturbance(MHDSimulation *sim, double intensity);
void            mhd_print_status(MHDSimulation *sim);

/* mhd_visualization.c - Visualization and export functions */
void            mhd_export_to_binary(MHDSimulation *sim, const char *filename);
void            mhd_export_to_hdf5(MHDSimulation *sim, const char *filename);
void            mhd_export_slice_data(MHDSimulation *sim, int slice_dim, int slice_pos, const char *filename);
void            mhd_export_field_data(MHDSimulation *sim, const char *field_name, const char *filename);

#endif /* MHD_H */
