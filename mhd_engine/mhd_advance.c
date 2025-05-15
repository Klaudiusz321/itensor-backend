/**
 * mhd_advanced.c - Advanced simulation options for MHD simulation
 *
 * Implements specialized and advanced features for MHD simulations including
 * turbulence, disturbances, dynamic changes, particle interactions, and
 * spatial gradients.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mhd.h"

/**
 * Set the simulation mode
 */
void mhd_set_simulation_mode(MHDSimulation *sim, SimulationMode mode) {
    if (!sim) return;
    sim->mode = mode;
}

/**
 * Apply random disturbances to the simulation
 * Adds random perturbations to the grid according to intensity.
 */
void mhd_apply_disturbances(MHDSimulation *sim, double intensity) {
    if (!sim || intensity <= 0.0) return;
    sim->random_disturbance_intensity = intensity;
    mhd_add_random_disturbance(sim, intensity);
}

/**
 * Apply dynamic (time-dependent) changes to the simulation
 */
void mhd_apply_dynamic_changes(MHDSimulation *sim) {
    if (!sim) return;
    double t = sim->current_time;
    /* Example oscillation parameters, could be configured */
    double freq = 0.2;
    double amp  = 0.1;
    double osc  = amp * sin(2.0 * PI * freq * t);
    for (int i = 1; i < sim->grid_size_x-1; ++i) {
        for (int j = 1; j < sim->grid_size_y-1; ++j) {
            for (int k = 1; k < sim->grid_size_z-1; ++k) {
                GridCell *c = &sim->grid[i][j][k];
                c->velocity.x += osc;
                c->velocity.y += amp * cos(2.0 * PI * freq * t);
                c->velocity.z += amp * sin(2.0 * PI * freq * t + PI/4);
            }
        }
    }
}

/**
 * Enable or disable turbulence flag
 */


/**
 * Apply turbulence structures to the velocity field
 */
void mhd_apply_turbulence(MHDSimulation *sim, double intensity) {
    if (!sim || intensity <= 0.0) return;
    sim->turbulence_intensity = intensity;
    int num = (int)(5.0 * intensity);
    if (num < 1) num = 1;
    for (int v = 0; v < num; ++v) {
        double cx = mhd_random_value(0.1, 0.9);
        double cy = mhd_random_value(0.1, 0.9);
        double cz = mhd_random_value(0.1, 0.9);
        double strength = mhd_random_value(0.01, 0.1) * intensity;
        for (int i = 1; i < sim->grid_size_x-1; ++i) {
            for (int j = 1; j < sim->grid_size_y-1; ++j) {
                for (int k = 1; k < sim->grid_size_z-1; ++k) {
                    double x_norm = (double)i / sim->grid_size_x;
                    double y_norm = (double)j / sim->grid_size_y;
                    double z_norm = (double)k / sim->grid_size_z;
                    double dx = x_norm - cx;
                    double dy = y_norm - cy;
                    double dz = z_norm - cz;
                    double dist = sqrt(dx*dx + dy*dy + dz*dz);
                    double radius = mhd_random_value(0.05, 0.2);
                    if (dist < radius) {
                        double factor = strength * (1.0 - dist / radius);
                        GridCell *c = &sim->grid[i][j][k];
                        switch (v % 3) {
                            case 0:
                                c->velocity.x += factor * dy * 10.0;
                                c->velocity.y -= factor * dx * 10.0;
                                break;
                            case 1:
                                c->velocity.y += factor * dz * 10.0;
                                c->velocity.z -= factor * dy * 10.0;
                                break;
                            case 2:
                                c->velocity.x -= factor * dz * 10.0;
                                c->velocity.z += factor * dx * 10.0;
                                break;
                        }
                    }
                }
            }
        }
    }
}

/**
 * Enable or disable particle interaction flag
 */
void mhd_enable_particle_interaction(MHDSimulation *sim, int enabled) {
    if (!sim) return;
    sim->use_particle_interaction = (enabled != 0);
}

/**
 * Apply particle interaction effects (simplified Lorentz force)
 */
void mhd_apply_particle_interaction(MHDSimulation *sim, bool enable) {
    if (!sim) return;
    sim->use_particle_interaction = enable;
    if (!enable) return;
    for (int i = 1; i < sim->grid_size_x-1; ++i) {
        for (int j = 1; j < sim->grid_size_y-1; ++j) {
            for (int k = 1; k < sim->grid_size_z-1; ++k) {
                GridCell *c = &sim->grid[i][j][k];
                double lx = c->velocity.y * c->magnetic.z - c->velocity.z * c->magnetic.y;
                double ly = c->velocity.z * c->magnetic.x - c->velocity.x * c->magnetic.z;
                double lz = c->velocity.x * c->magnetic.y - c->velocity.y * c->magnetic.x;
                double scale = 0.01;
                c->velocity.x += scale * lx;
                c->velocity.y += scale * ly;
                c->velocity.z += scale * lz;
            }
        }
    }
}

/**
 * Enable or disable spatial gradients flag
 */
void mhd_enable_spatial_gradients(MHDSimulation *sim, int enabled) {
    if (!sim) return;
    sim->use_spatial_gradients = (enabled != 0);
}

/**
 * Apply spatial gradients of density, temperature, and magnetic field
 */
void mhd_apply_spatial_gradients(MHDSimulation *sim, bool enable) {
    if (!sim || !enable) return;
    double dg = 0.5;
    double tg = 1.0;
    double mg = 0.3;
    double bd = sim->initial_density;
    double bt = sim->initial_temperature;
    Vector3D bm = sim->initial_magnetic_field;
    for (int i = 0; i < sim->grid_size_x; ++i) {
        for (int j = 0; j < sim->grid_size_y; ++j) {
            for (int k = 0; k < sim->grid_size_z; ++k) {
                double x = (double)i / sim->grid_size_x;
                double y = (double)j / sim->grid_size_y;
                double z = (double)k / sim->grid_size_z;
                GridCell *c = &sim->grid[i][j][k];
                c->density     = bd * (1.0 + dg * x);
                c->temperature = bt * (1.0 + tg * z);
                c->pressure    = c->density * c->temperature;
                c->magnetic.x  = bm.x * (1.0 + mg * y);
                c->magnetic.y  = bm.y * (1.0 + mg * y);
                c->magnetic.z  = bm.z * (1.0 + mg * y);
            }
        }
    }
}

/**
 * Set the intensity for random disturbances
 */
void mhd_set_random_disturbance_intensity(MHDSimulation *sim, double intensity) {
    if (!sim) return;
    if (intensity < 0.0) intensity = fabs(intensity);
    sim->random_disturbance_intensity = intensity;
}

void mhd_set_turbulence_strength(MHDSimulation *sim, double strength) {
    if (!sim) return;
    sim->turbulence_intensity = strength;
}

void mhd_enable_disturbance(MHDSimulation *sim, int enabled) {
    if (!sim) return;
    sim->disturbance_enabled = (enabled != 0);
}

void mhd_set_disturbance_frequency(MHDSimulation *sim, double freq) {
    if (!sim) return;
    sim->disturbance_frequency = freq;
}

void mhd_set_disturbance_magnitude(MHDSimulation *sim, double mag) {
    if (!sim) return;
    sim->disturbance_magnitude = mag;
}
void mhd_enable_turbulence(MHDSimulation *sim, int enabled) {
    if (!sim) return;
    sim->turbulence_enabled = (enabled != 0);
}

void mhd_enable_random_disturbances(MHDSimulation *sim, int enabled) {
    if (!sim) return;
    sim->random_disturbances_enabled = (enabled != 0);
}

void mhd_set_random_disturbance_magnitude(MHDSimulation *sim, double mag) {
    if (!sim) return;
    sim->random_disturbance_magnitude = mag;
}