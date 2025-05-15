/**
 * mhd_conditions.c - Boundary conditions and dynamic field support for MHD simulation
 *
 * Implements various boundary conditions (open, closed, reflective, periodic, custom)
 * and provides setters for dynamic magnetic field parameters.
 */

#include "mhd.h"

/*
 * Set the boundary conditions type for the simulation
 */
void mhd_set_boundary_type(MHDSimulation *sim, BoundaryType type) {
    if (!sim) return;
    sim->boundary_type = type;
}

/*
 * Register a custom boundary function and set boundary type to CUSTOM
 */
void mhd_set_custom_boundaries(MHDSimulation *sim,
                               CustomBoundaryFunc func)
{
    if (!sim) return;
    sim->custom_boundary_func = func;
    sim->boundary_type        = BC_CUSTOM;
}

/**
 * Zastosuj warunki brzegowe zgodnie z aktualnym typem w sim->boundary_type
 */
void mhd_apply_boundary_conditions(MHDSimulation *sim, GridCell ***buf) {
    if (!sim || !buf) return;

    switch (sim->boundary_type) {
        case BC_OPEN:
            mhd_set_open_boundaries(sim, buf);
            break;
        case BC_CLOSED:
            mhd_set_closed_boundaries(sim, buf);
            break;
        case BC_REFLECTIVE:
            mhd_set_reflective_boundaries(sim, buf);
            break;
        case BC_PERIODIC:
            mhd_set_periodic_boundaries(sim, buf);
            break;
        case BC_CUSTOM:
            if (sim->custom_boundary_func) {
                sim->custom_boundary_func(sim, buf);
            }
            break;
        default:
            // Jeśli typ nieznany, używamy periodic jako fallback
            mhd_set_periodic_boundaries(sim, buf);
            break;
    }
}

/**
 * Set open (zero-gradient) boundary conditions
 */
void mhd_set_open_boundaries(MHDSimulation *sim, GridCell ***buf) {
    if (!sim) return;
    int nx = sim->grid_size_x;
    int ny = sim->grid_size_y;
    int nz = sim->grid_size_z;

    /* X boundaries */
    for (int j = 0; j < ny; j++)
    for (int k = 0; k < nz; k++) {
        buf[0][j][k]    = buf[1][j][k];
        buf[nx-1][j][k] = buf[nx-2][j][k];
    }
    /* Y boundaries */
    for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) {
        buf[i][0][k]    = buf[i][1][k];
        buf[i][ny-1][k] = buf[i][ny-2][k];
    }
    /* Z boundaries */
    if (nz > 1) {
        for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            buf[i][j][0]    = buf[i][j][1];
            buf[i][j][nz-1] = buf[i][j][nz-2];
        }
    }
}

/**
 * Set closed (impenetrable) boundary conditions: zero normal velocity,
 * zero-gradient for other quantities.
 */
void mhd_set_closed_boundaries(MHDSimulation *sim, GridCell ***buf) {
    if (!sim) return;
    int nx = sim->grid_size_x;
    int ny = sim->grid_size_y;
    int nz = sim->grid_size_z;

    /* X boundaries */
    for (int j = 0; j < ny; j++)
    for (int k = 0; k < nz; k++) {
        GridCell gc = buf[1][j][k];
        gc.velocity.x = 0.0;
        buf[0][j][k]    = gc;
        gc = buf[nx-2][j][k];
        gc.velocity.x = 0.0;
        buf[nx-1][j][k] = gc;
    }
    /* Y boundaries */
    for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) {
        GridCell gc = buf[i][1][k];
        gc.velocity.y = 0.0;
        buf[i][0][k]    = gc;
        gc = buf[i][ny-2][k];
        gc.velocity.y = 0.0;
        buf[i][ny-1][k] = gc;
    }
    /* Z boundaries */
    if (nz > 1) {
        for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            GridCell gc = buf[i][j][1];
            gc.velocity.z = 0.0;
            buf[i][j][0]    = gc;
            gc = buf[i][j][nz-2];
            gc.velocity.z = 0.0;
            buf[i][j][nz-1] = gc;
        }
    }
}

/**
 * Set reflective boundary conditions: reverse normal velocity,
 * invert tangential magnetic field.
 */
void mhd_set_reflective_boundaries(MHDSimulation *sim, GridCell ***buf) {
    if (!sim) return;
    int nx = sim->grid_size_x;
    int ny = sim->grid_size_y;
    int nz = sim->grid_size_z;

    /* X boundaries */
    for (int j = 0; j < ny; j++)
    for (int k = 0; k < nz; k++) {
        GridCell in = buf[1][j][k];
        in.velocity.x = -in.velocity.x;
        in.magnetic.y = -in.magnetic.y;
        in.magnetic.z = -in.magnetic.z;
        buf[0][j][k] = in;

        in = buf[nx-2][j][k];
        in.velocity.x = -in.velocity.x;
        in.magnetic.y = -in.magnetic.y;
        in.magnetic.z = -in.magnetic.z;
        buf[nx-1][j][k] = in;
    }
    /* Y boundaries */
    for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) {
        GridCell in = buf[i][1][k];
        in.velocity.y = -in.velocity.y;
        in.magnetic.x = -in.magnetic.x;
        in.magnetic.z = -in.magnetic.z;
        buf[i][0][k] = in;

        in = buf[i][ny-2][k];
        in.velocity.y = -in.velocity.y;
        in.magnetic.x = -in.magnetic.x;
        in.magnetic.z = -in.magnetic.z;
        buf[i][ny-1][k] = in;
    }
    /* Z boundaries */
    if (nz > 1) {
        for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            GridCell in = buf[i][j][1];
            in.velocity.z = -in.velocity.z;
            in.magnetic.x = -in.magnetic.x;
            in.magnetic.y = -in.magnetic.y;
            buf[i][j][0] = in;

            in = buf[i][j][nz-2];
            in.velocity.z = -in.velocity.z;
            in.magnetic.x = -in.magnetic.x;
            in.magnetic.y = -in.magnetic.y;
            buf[i][j][nz-1] = in;
        }
    }
}

/**
 * Set periodic boundary conditions
 */
void mhd_set_periodic_boundaries(MHDSimulation *sim, GridCell ***buf) {
    if (!sim) return;
    int nx = sim->grid_size_x;
    int ny = sim->grid_size_y;
    int nz = sim->grid_size_z;

    /* X boundaries */
    for (int j = 0; j < ny; j++)
    for (int k = 0; k < nz; k++) {
        buf[0][j][k]    = buf[nx-2][j][k];
        buf[nx-1][j][k] = buf[1][j][k];
    }
    /* Y boundaries */
    for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) {
        buf[i][0][k]    = buf[i][ny-2][k];
        buf[i][ny-1][k] = buf[i][1][k];
    }
    /* Z boundaries */
    if (nz > 1) {
        for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            buf[i][j][0]    = buf[i][j][nz-2];
            buf[i][j][nz-1] = buf[i][j][1];
        }
    }
}

/*
 * --- Dynamic Field Support ---
 */

/**
 * Enable or disable time‐varying magnetic field
 */
void mhd_enable_dynamic_field(MHDSimulation *sim, int enabled) {
    if (!sim) return;
    sim->dynamic_field_enabled = (enabled != 0);
}

/**
 * Set the dynamic‐field variation type
 */
void mhd_set_dynamic_field_type(MHDSimulation *sim, int type) {
    if (!sim) return;
    sim->dynamic_field_type = type;
}

/**
 * Set the amplitude of the dynamic magnetic field
 */
void mhd_set_dynamic_field_amplitude(MHDSimulation *sim, double amplitude) {
    if (!sim) return;
    sim->dynamic_field_amplitude = amplitude;
}

/**
 * Set the frequency of the dynamic magnetic field
 */
void mhd_set_dynamic_field_frequency(MHDSimulation *sim, double frequency) {
    if (!sim) return;
    sim->dynamic_field_frequency = frequency;
}

void mhd_enable_spatial_gradient(MHDSimulation *sim,
                                 const char     *field_name,
                                 int             enabled)
{
    int flag = (enabled != 0);

    if (strcmp(field_name, "density") == 0) {
        sim->gradient_density_enabled = flag;
    }
    else if (strcmp(field_name, "velocity") == 0) {
        sim->gradient_velocity_enabled = flag;
    }
    else if (strcmp(field_name, "magnetic_field") == 0) {
        sim->gradient_magnetic_enabled = flag;
    }
    else if (strcmp(field_name, "temperature") == 0) {
        sim->gradient_temperature_enabled = flag;
    }
    else if (strcmp(field_name, "pressure") == 0) {
        sim->gradient_pressure_enabled = flag;
    }
    // … analogicznie dla innych pól …
    else {
        // nieznane pole – możesz logować ostrzeżenie albo ignorować
    }
}


