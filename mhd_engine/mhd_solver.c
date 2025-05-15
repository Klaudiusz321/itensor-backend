/**
 * mhd_solver.c - Numerical solvers for MHD simulation
 * 
 * Implements Euler, Runge-Kutta 2nd order, and Runge-Kutta 4th order 
 * numerical methods for solving MHD equations.
 */

#include "mhd.h"
#include <math.h>

/* Constants for numerical stability - już zdefiniowane w mhd.h */
#define MIN_DENSITY 1e-6
#define MAX_DENSITY 1e6
#define MIN_PRESSURE 1e-6
#define MAX_PRESSURE 1e6
#define MIN_TEMPERATURE 1e-6
#define MAX_TEMPERATURE 1e6

/* Helper macros for min and max */
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

/**
 * Set the numerical method used for simulation
 */
void mhd_set_numerical_method(MHDSimulation *sim, NumericalMethod method) {
    if (!sim) return;
    sim->method = method;
}

/**
 * Calculate field derivatives for MHD equations, including advection of ρ
 */
void mhd_update_derivatives(MHDSimulation *sim,
                            GridCell ***source,
                            GridCell ***derivatives)
{
    if (!sim || !source || !derivatives) return;

    int nx = sim->grid_size_x;
    int ny = sim->grid_size_y;
    int nz = sim->grid_size_z;
    
    // Użyj bezpiecznych wartości dla parametrów fizycznych
    double eta   = fmin(sim->magnetic_viscosity, 0.01);
    double sigma = fmin(sim->magnetic_conductivity, 0.1);
    double cs2   = fmin(sim->sound_speed * sim->sound_speed, 0.1);

    /* apply BC to the source buffer */
    mhd_apply_boundary_conditions(sim, source);

    // Reset derivatives to zero to avoid accumulating garbage values
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                derivatives[i][j][k].density = 0.0;
                derivatives[i][j][k].pressure = 0.0;
                derivatives[i][j][k].temperature = 0.0;
                derivatives[i][j][k].velocity.x = 0.0;
                derivatives[i][j][k].velocity.y = 0.0;
                derivatives[i][j][k].velocity.z = 0.0;
                derivatives[i][j][k].magnetic.x = 0.0;
                derivatives[i][j][k].magnetic.y = 0.0;
                derivatives[i][j][k].magnetic.z = 0.0;
            }
        }
    }

    // Helper function to safely compute differences
    double safe_diff(double a, double b) {
        // Sprawdź czy wartości są skończone
        if (!isfinite(a) || !isfinite(b)) {
            return 0.0;  // Zwróć 0 dla nieskończoności lub NaN
        }
        
        double diff = a - b;
        // Limit maximum difference to avoid extreme gradients
        double max_diff = 1.0;
        if (diff > max_diff) return max_diff;
        if (diff < -max_diff) return -max_diff;
        return diff;
    }
    
    // Helper function to safely get a value with limits
    double safe_value(double value, double min_val, double max_val) {
        if (!isfinite(value)) return min_val;  // Default to min for NaN/Inf
        if (value < min_val) return min_val;
        if (value > max_val) return max_val;
        return value;
    }

    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
            for (int k = 1; k < nz - 1; k++) {
                GridCell *c  = &source[i][j][k];
                
                // Apply constraints to ensure physical values
                enforce_physical_constraints(c);
                
                // Ensure density is positive and within limits
                c->density = safe_value(c->density, MIN_DENSITY, MAX_DENSITY);
                
                // Ensure pressure is positive and within limits
                c->pressure = safe_value(c->pressure, MIN_PRESSURE, MAX_PRESSURE);
                
                // Ensure temperature is positive and within limits
                c->temperature = safe_value(c->temperature, MIN_TEMPERATURE, MAX_TEMPERATURE);
                
                // Limit velocity components
                c->velocity.x = safe_value(c->velocity.x, -MAX_VELOCITY * 0.5, MAX_VELOCITY * 0.5);
                c->velocity.y = safe_value(c->velocity.y, -MAX_VELOCITY * 0.5, MAX_VELOCITY * 0.5);
                c->velocity.z = safe_value(c->velocity.z, -MAX_VELOCITY * 0.5, MAX_VELOCITY * 0.5);
                
                // Limit magnetic field components
                c->magnetic.x = safe_value(c->magnetic.x, -MAX_MAGNETIC * 0.5, MAX_MAGNETIC * 0.5);
                c->magnetic.y = safe_value(c->magnetic.y, -MAX_MAGNETIC * 0.5, MAX_MAGNETIC * 0.5);
                c->magnetic.z = safe_value(c->magnetic.z, -MAX_MAGNETIC * 0.5, MAX_MAGNETIC * 0.5);
                
                GridCell *xp = &source[i+1][j][k];
                GridCell *xm = &source[i-1][j][k];
                GridCell *yp = &source[i][j+1][k];
                GridCell *ym = &source[i][j-1][k];
                GridCell *zp = &source[i][j][k+1];
                GridCell *zm = &source[i][j][k-1];

                // Apply constraints to neighboring cells too
                enforce_physical_constraints(xp);
                enforce_physical_constraints(xm);
                enforce_physical_constraints(yp);
                enforce_physical_constraints(ym);
                enforce_physical_constraints(zp);
                enforce_physical_constraints(zm);

                // 1) divergence of v and B - use safe differences
                double dvx = safe_diff(xp->velocity.x, xm->velocity.x) * 0.5;
                double dvy = safe_diff(yp->velocity.y, ym->velocity.y) * 0.5;
                double dvz = safe_diff(zp->velocity.z, zm->velocity.z) * 0.5;
                double div_v = dvx + dvy + dvz;
                
                // Limit maximum divergence
                double max_div = 0.5;
                if (fabs(div_v) > max_div) {
                    div_v = (div_v > 0) ? max_div : -max_div;
                }

                double dBx = safe_diff(xp->magnetic.x, xm->magnetic.x) * 0.5;
                double dBy = safe_diff(yp->magnetic.y, ym->magnetic.y) * 0.5;
                double dBz = safe_diff(zp->magnetic.z, zm->magnetic.z) * 0.5;
                double div_B = dBx + dBy + dBz;
                
                // Limit maximum magnetic divergence
                if (fabs(div_B) > max_div) {
                    div_B = (div_B > 0) ? max_div : -max_div;
                }

                // 2) advection of density - use safe differences
                double drho_dx = safe_diff(xp->density, xm->density) * 0.5;
                double drho_dy = safe_diff(yp->density, ym->density) * 0.5;
                double drho_dz = safe_diff(zp->density, zm->density) * 0.5;
                
                // Limit velocity components for advection calculation
                double vx_safe = c->velocity.x;
                double vy_safe = c->velocity.y;
                double vz_safe = c->velocity.z;
                
                double max_vel = 0.5;
                if (fabs(vx_safe) > max_vel) vx_safe = (vx_safe > 0) ? max_vel : -max_vel;
                if (fabs(vy_safe) > max_vel) vy_safe = (vy_safe > 0) ? max_vel : -max_vel;
                if (fabs(vz_safe) > max_vel) vz_safe = (vz_safe > 0) ? max_vel : -max_vel;
                
                double adv_rho = vx_safe * drho_dx + vy_safe * drho_dy + vz_safe * drho_dz;
                
                // Limit advection term
                double max_adv_rho = 0.5 * c->density;
                if (fabs(adv_rho) > max_adv_rho) {
                    adv_rho = (adv_rho > 0) ? max_adv_rho : -max_adv_rho;
                }

                // 3) advection of v - use safe differences and limited velocities
                double vx_adv = vx_safe * dvx
                              + vy_safe * (safe_diff(xp->velocity.x, xm->velocity.x) * 0.5)
                              + vz_safe * (safe_diff(zp->velocity.x, zm->velocity.x) * 0.5);
                              
                double vy_adv = vx_safe * (safe_diff(xp->velocity.y, xm->velocity.y) * 0.5)
                              + vy_safe * dvy
                              + vz_safe * (safe_diff(zp->velocity.y, zm->velocity.y) * 0.5);
                              
                double vz_adv = vx_safe * (safe_diff(xp->velocity.z, xm->velocity.z) * 0.5)
                              + vy_safe * (safe_diff(yp->velocity.z, ym->velocity.z) * 0.5)
                              + vz_safe * dvz;

                // Limit advection terms
                double max_adv = 0.5;
                if (fabs(vx_adv) > max_adv) vx_adv = (vx_adv > 0) ? max_adv : -max_adv;
                if (fabs(vy_adv) > max_adv) vy_adv = (vy_adv > 0) ? max_adv : -max_adv;
                if (fabs(vz_adv) > max_adv) vz_adv = (vz_adv > 0) ? max_adv : -max_adv;

                // 4) curl(v×B) - use safe differences and limited field values
                // Helper function to safely get magnetic field components
                double safe_b(double b) {
                    double max_b = 0.5;
                    if (!isfinite(b)) return 0.0;
                    return (fabs(b) > max_b) ? ((b > 0) ? max_b : -max_b) : b;
                }
                
                // Helper function to safely get velocity components
                double safe_v(double v) {
                    double max_v = 0.5;
                    if (!isfinite(v)) return 0.0;
                    return (fabs(v) > max_v) ? ((v > 0) ? max_v : -max_v) : v;
                }
                
                double curl_vB_x = ((safe_v(yp->velocity.z) * safe_b(yp->magnetic.y))
                                   - (safe_v(ym->velocity.z) * safe_b(ym->magnetic.y))) * 0.5
                                 - ((safe_v(zp->velocity.y) * safe_b(zp->magnetic.z))
                                   - (safe_v(zm->velocity.y) * safe_b(zm->magnetic.z))) * 0.5;
                                   
                double curl_vB_y = ((safe_v(zp->velocity.x) * safe_b(zp->magnetic.z))
                                   - (safe_v(zm->velocity.x) * safe_b(zm->magnetic.z))) * 0.5
                                 - ((safe_v(xp->velocity.z) * safe_b(xp->magnetic.x))
                                   - (safe_v(xm->velocity.z) * safe_b(xm->magnetic.x))) * 0.5;
                                   
                double curl_vB_z = ((safe_v(xp->velocity.y) * safe_b(xp->magnetic.x))
                                   - (safe_v(xm->velocity.y) * safe_b(xm->magnetic.x))) * 0.5
                                 - ((safe_v(yp->velocity.x) * safe_b(yp->magnetic.y))
                                   - (safe_v(ym->velocity.x) * safe_b(ym->magnetic.y))) * 0.5;

                // Limit curl terms
                double max_curl = 0.01;
                if (fabs(curl_vB_x) > max_curl) curl_vB_x = (curl_vB_x > 0) ? max_curl : -max_curl;
                if (fabs(curl_vB_y) > max_curl) curl_vB_y = (curl_vB_y > 0) ? max_curl : -max_curl;
                if (fabs(curl_vB_z) > max_curl) curl_vB_z = (curl_vB_z > 0) ? max_curl : -max_curl;

                // 5) Lorentz force - use limited values
                double curl_B_x = safe_diff(dBy, dBz);
                double curl_B_y = safe_diff(dBz, dBx);
                double curl_B_z = safe_diff(dBx, dBy);
                
                // Use a very small sigma value for stability
                double safe_sigma = fmin(sigma, 0.005);
                
                double lorentz_x = (curl_B_y * safe_b(c->magnetic.z) - curl_B_z * safe_b(c->magnetic.y)) * safe_sigma;
                double lorentz_y = (curl_B_z * safe_b(c->magnetic.x) - curl_B_x * safe_b(c->magnetic.z)) * safe_sigma;
                double lorentz_z = (curl_B_x * safe_b(c->magnetic.y) - curl_B_y * safe_b(c->magnetic.x)) * safe_sigma;

                // Limit Lorentz force
                double max_lorentz = 0.01;
                if (fabs(lorentz_x) > max_lorentz) lorentz_x = (lorentz_x > 0) ? max_lorentz : -max_lorentz;
                if (fabs(lorentz_y) > max_lorentz) lorentz_y = (lorentz_y > 0) ? max_lorentz : -max_lorentz;
                if (fabs(lorentz_z) > max_lorentz) lorentz_z = (lorentz_z > 0) ? max_lorentz : -max_lorentz;

                // 6) diffusion of B - use limited values and safe differences
                // Use a very small eta value for stability
                double safe_eta = fmin(eta, 0.0005);
                
                double lap_Bx = safe_b(xp->magnetic.x) + safe_b(xm->magnetic.x)
                              + safe_b(yp->magnetic.x) + safe_b(ym->magnetic.x)
                              + safe_b(zp->magnetic.x) + safe_b(zm->magnetic.x)
                              - 6.0 * safe_b(c->magnetic.x);
                              
                double lap_By = safe_b(xp->magnetic.y) + safe_b(xm->magnetic.y)
                              + safe_b(yp->magnetic.y) + safe_b(ym->magnetic.y)
                              + safe_b(zp->magnetic.y) + safe_b(zm->magnetic.y)
                              - 6.0 * safe_b(c->magnetic.y);
                              
                double lap_Bz = safe_b(xp->magnetic.z) + safe_b(xm->magnetic.z)
                              + safe_b(yp->magnetic.z) + safe_b(ym->magnetic.z)
                              + safe_b(zp->magnetic.z) + safe_b(zm->magnetic.z)
                              - 6.0 * safe_b(c->magnetic.z);
                              
                double diff_Bx = safe_eta * lap_Bx;
                double diff_By = safe_eta * lap_By;
                double diff_Bz = safe_eta * lap_Bz;

                // Limit diffusion terms
                double max_diff = 0.01;
                if (fabs(diff_Bx) > max_diff) diff_Bx = (diff_Bx > 0) ? max_diff : -max_diff;
                if (fabs(diff_By) > max_diff) diff_By = (diff_By > 0) ? max_diff : -max_diff;
                if (fabs(diff_Bz) > max_diff) diff_Bz = (diff_Bz > 0) ? max_diff : -max_diff;

                // write derivatives
                GridCell *d = &derivatives[i][j][k];

                // Calculate derivatives with safety checks to prevent division by zero
                double safe_density = MAX(c->density, MIN_DENSITY);
                
                // Pressure gradients with safety limits
                double dp_dx = safe_diff(xp->pressure, xm->pressure) * 0.5;
                double dp_dy = safe_diff(yp->pressure, ym->pressure) * 0.5;
                double dp_dz = safe_diff(zp->pressure, zm->pressure) * 0.5;
                
                // Limit pressure gradients
                double max_dp = 0.05;
                if (fabs(dp_dx) > max_dp) dp_dx = (dp_dx > 0) ? max_dp : -max_dp;
                if (fabs(dp_dy) > max_dp) dp_dy = (dp_dy > 0) ? max_dp : -max_dp;
                if (fabs(dp_dz) > max_dp) dp_dz = (dp_dz > 0) ? max_dp : -max_dp;

                // Set final derivatives with strict limits
                d->density = MAX(-0.05 * safe_density, MIN(-adv_rho - c->density * div_v, 0.05 * safe_density));
                
                d->velocity.x = MAX(-0.05, MIN(-vx_adv - dp_dx / safe_density + lorentz_x / safe_density, 0.05));
                d->velocity.y = MAX(-0.05, MIN(-vy_adv - dp_dy / safe_density + lorentz_y / safe_density, 0.05));
                d->velocity.z = MAX(-0.05, MIN(-vz_adv - dp_dz / safe_density + lorentz_z / safe_density, 0.05));
                
                d->magnetic.x = MAX(-0.01, MIN(curl_vB_x + diff_Bx, 0.01));
                d->magnetic.y = MAX(-0.01, MIN(curl_vB_y + diff_By, 0.01));
                d->magnetic.z = MAX(-0.01, MIN(curl_vB_z + diff_Bz, 0.01));
                
                // Apply divergence cleaning to magnetic field derivatives
                apply_divergence_cleaning(d, div_B, sim->time_step);
                
                // Simple pressure and temperature updates with stricter limits
                d->pressure = MAX(-0.05 * c->pressure, MIN(-cs2 * c->density * div_v, 0.05 * c->pressure));
                d->temperature = MAX(-0.05 * c->temperature, MIN(0.0, 0.05 * c->temperature));
                
                // Final sanity check - zero out any non-finite values
                if (!isfinite(d->density))     d->density = 0.0;
                if (!isfinite(d->pressure))    d->pressure = 0.0;
                if (!isfinite(d->temperature)) d->temperature = 0.0;
                if (!isfinite(d->velocity.x))  d->velocity.x = 0.0;
                if (!isfinite(d->velocity.y))  d->velocity.y = 0.0;
                if (!isfinite(d->velocity.z))  d->velocity.z = 0.0;
                if (!isfinite(d->magnetic.x))  d->magnetic.x = 0.0;
                if (!isfinite(d->magnetic.y))  d->magnetic.y = 0.0;
                if (!isfinite(d->magnetic.z))  d->magnetic.z = 0.0;
            }
        }
    }
}

/**
 * Single Euler step
 */
void mhd_solver_euler_step(MHDSimulation *sim) {
    if (!sim) return;

    // 1) compute derivatives into temp_grid (with BC inside)
    mhd_update_derivatives(sim, sim->grid, sim->temp_grid);

    // 2) update
    for (int i = 1; i < sim->grid_size_x-1; i++)
    for (int j = 1; j < sim->grid_size_y-1; j++)
    for (int k = 1; k < sim->grid_size_z-1; k++) {
        GridCell *g = &sim->grid[i][j][k];
        GridCell *d = &sim->temp_grid[i][j][k];
        
        // Apply updates with stability factor
        g->density    += sim->time_step * d->density * STABILITY_FACTOR;
        g->pressure   += sim->time_step * d->pressure * STABILITY_FACTOR;
        g->temperature+= sim->time_step * d->temperature * STABILITY_FACTOR;
        g->velocity.x += sim->time_step * d->velocity.x * STABILITY_FACTOR;
        g->velocity.y += sim->time_step * d->velocity.y * STABILITY_FACTOR;
        g->velocity.z += sim->time_step * d->velocity.z * STABILITY_FACTOR;
        g->magnetic.x += sim->time_step * d->magnetic.x * STABILITY_FACTOR;
        g->magnetic.y += sim->time_step * d->magnetic.y * STABILITY_FACTOR;
        g->magnetic.z += sim->time_step * d->magnetic.z * STABILITY_FACTOR;
        
        // Enforce physical constraints after update
        enforce_physical_constraints(g);
    }
}

/**
 * Single RK2 (midpoint) step
 */
void mhd_solver_rk2_step(MHDSimulation *sim) {
    if (!sim) return;

    GridCell ***k1  = sim->temp_grid;
    GridCell ***mid = sim->temp_grid2;

    // k1
    mhd_update_derivatives(sim, sim->grid, k1);

    // midpoint state: grid + dt/2 * k1
    for (int i = 1; i < sim->grid_size_x-1; i++)
    for (int j = 1; j < sim->grid_size_y-1; j++)
    for (int k = 1; k < sim->grid_size_z-1; k++) {
        GridCell *g = &sim->grid[i][j][k];
        GridCell tmp = *g;
        GridCell *d1 = &k1[i][j][k];
        
        // Apply updates with stability factor
        tmp.density     += 0.5*sim->time_step * d1->density * STABILITY_FACTOR;
        tmp.pressure    += 0.5*sim->time_step * d1->pressure * STABILITY_FACTOR;
        tmp.temperature += 0.5*sim->time_step * d1->temperature * STABILITY_FACTOR;
        tmp.velocity.x  += 0.5*sim->time_step * d1->velocity.x * STABILITY_FACTOR;
        tmp.velocity.y  += 0.5*sim->time_step * d1->velocity.y * STABILITY_FACTOR;
        tmp.velocity.z  += 0.5*sim->time_step * d1->velocity.z * STABILITY_FACTOR;
        tmp.magnetic.x  += 0.5*sim->time_step * d1->magnetic.x * STABILITY_FACTOR;
        tmp.magnetic.y  += 0.5*sim->time_step * d1->magnetic.y * STABILITY_FACTOR;
        tmp.magnetic.z  += 0.5*sim->time_step * d1->magnetic.z * STABILITY_FACTOR;
        
        // Enforce physical constraints on midpoint state
        enforce_physical_constraints(&tmp);
        
        mid[i][j][k] = tmp;
    }

    // k2
    mhd_update_derivatives(sim, mid, k1);

    // final update
    for (int i = 1; i < sim->grid_size_x-1; i++)
    for (int j = 1; j < sim->grid_size_y-1; j++)
    for (int k = 1; k < sim->grid_size_z-1; k++) {
        GridCell *g = &sim->grid[i][j][k];
        GridCell *d2= &k1[i][j][k];
        
        // Apply updates with stability factor
        g->density    += sim->time_step * d2->density * STABILITY_FACTOR;
        g->pressure   += sim->time_step * d2->pressure * STABILITY_FACTOR;
        g->temperature+= sim->time_step * d2->temperature * STABILITY_FACTOR;
        g->velocity.x += sim->time_step * d2->velocity.x * STABILITY_FACTOR;
        g->velocity.y += sim->time_step * d2->velocity.y * STABILITY_FACTOR;
        g->velocity.z += sim->time_step * d2->velocity.z * STABILITY_FACTOR;
        g->magnetic.x += sim->time_step * d2->magnetic.x * STABILITY_FACTOR;
        g->magnetic.y += sim->time_step * d2->magnetic.y * STABILITY_FACTOR;
        g->magnetic.z += sim->time_step * d2->magnetic.z * STABILITY_FACTOR;
        
        // Enforce physical constraints after update
        enforce_physical_constraints(g);
    }
}

/**
 * Single RK4 step
 */
void mhd_solver_rk4_step(MHDSimulation *sim) {
    if (!sim) return;

    GridCell ***k1  = sim->temp_grid1;
    GridCell ***k2  = sim->temp_grid2;
    GridCell ***k3  = sim->temp_grid3;
    GridCell ***k4  = sim->temp_grid4;
    GridCell ***tmp = sim->tmp_grid;

    // k1
    mhd_update_derivatives(sim, sim->grid, k1);

    // k2: state = grid + dt/2*k1
    for (int i = 1; i < sim->grid_size_x-1; i++)
    for (int j = 1; j < sim->grid_size_y-1; j++)
    for (int k = 1; k < sim->grid_size_z-1; k++) {
        GridCell *g=&sim->grid[i][j][k];
        GridCell *d1=&k1[i][j][k];
        tmp[i][j][k] = *g;
        tmp[i][j][k].density     += 0.5*sim->time_step*d1->density;
        tmp[i][j][k].pressure    += 0.5*sim->time_step*d1->pressure;
        tmp[i][j][k].temperature += 0.5*sim->time_step*d1->temperature;
        tmp[i][j][k].velocity.x  += 0.5*sim->time_step*d1->velocity.x;
        tmp[i][j][k].velocity.y  += 0.5*sim->time_step*d1->velocity.y;
        tmp[i][j][k].velocity.z  += 0.5*sim->time_step*d1->velocity.z;
        tmp[i][j][k].magnetic.x  += 0.5*sim->time_step*d1->magnetic.x;
        tmp[i][j][k].magnetic.y  += 0.5*sim->time_step*d1->magnetic.y;
        tmp[i][j][k].magnetic.z  += 0.5*sim->time_step*d1->magnetic.z;
    }
    mhd_update_derivatives(sim, tmp, k2);

    // k3: state = grid + dt/2*k2
    for (int i = 1; i < sim->grid_size_x-1; i++)
    for (int j = 1; j < sim->grid_size_y-1; j++)
    for (int k = 1; k < sim->grid_size_z-1; k++) {
        GridCell *g=&sim->grid[i][j][k];
        GridCell *d2=&k2[i][j][k];
        tmp[i][j][k] = *g;
        tmp[i][j][k].density     += 0.5*sim->time_step*d2->density;
        tmp[i][j][k].pressure    += 0.5*sim->time_step*d2->pressure;
        tmp[i][j][k].temperature += 0.5*sim->time_step*d2->temperature;
        tmp[i][j][k].velocity.x  += 0.5*sim->time_step*d2->velocity.x;
        tmp[i][j][k].velocity.y  += 0.5*sim->time_step*d2->velocity.y;
        tmp[i][j][k].velocity.z  += 0.5*sim->time_step*d2->velocity.z;
        tmp[i][j][k].magnetic.x  += 0.5*sim->time_step*d2->magnetic.x;
        tmp[i][j][k].magnetic.y  += 0.5*sim->time_step*d2->magnetic.y;
        tmp[i][j][k].magnetic.z  += 0.5*sim->time_step*d2->magnetic.z;
    }
    mhd_update_derivatives(sim, tmp, k3);

    // k4: state = grid + dt*k3
    for (int i = 1; i < sim->grid_size_x-1; i++)
    for (int j = 1; j < sim->grid_size_y-1; j++)
    for (int k = 1; k < sim->grid_size_z-1; k++) {
        GridCell *g=&sim->grid[i][j][k];
        GridCell *d3=&k3[i][j][k];
        tmp[i][j][k] = *g;
        tmp[i][j][k].density     +=    sim->time_step*d3->density;
        tmp[i][j][k].pressure    +=    sim->time_step*d3->pressure;
        tmp[i][j][k].temperature +=    sim->time_step*d3->temperature;
        tmp[i][j][k].velocity.x  +=    sim->time_step*d3->velocity.x;
        tmp[i][j][k].velocity.y  +=    sim->time_step*d3->velocity.y;
        tmp[i][j][k].velocity.z  +=    sim->time_step*d3->velocity.z;
        tmp[i][j][k].magnetic.x  +=    sim->time_step*d3->magnetic.x;
        tmp[i][j][k].magnetic.y  +=    sim->time_step*d3->magnetic.y;
        tmp[i][j][k].magnetic.z  +=    sim->time_step*d3->magnetic.z;
    }
    mhd_update_derivatives(sim, tmp, k4);

    // final update
    for (int i = 1; i < sim->grid_size_x-1; i++)
    for (int j = 1; j < sim->grid_size_y-1; j++)
    for (int k = 1; k < sim->grid_size_z-1; k++) {
        GridCell *g=&sim->grid[i][j][k];
        GridCell *d1=&k1[i][j][k];
        GridCell *d2=&k2[i][j][k];
        GridCell *d3=&k3[i][j][k];
        GridCell *d4=&k4[i][j][k];
        g->density     += sim->time_step/6.0 *
                          (d1->density  + 2*d2->density  + 2*d3->density  + d4->density);
        g->pressure    += sim->time_step/6.0 *
                          (d1->pressure + 2*d2->pressure + 2*d3->pressure + d4->pressure);
        g->temperature += sim->time_step/6.0 *
                          (d1->temperature + 2*d2->temperature + 2*d3->temperature + d4->temperature);
        g->velocity.x  += sim->time_step/6.0 *
                          (d1->velocity.x + 2*d2->velocity.x + 2*d3->velocity.x + d4->velocity.x);
        g->velocity.y  += sim->time_step/6.0 *
                          (d1->velocity.y + 2*d2->velocity.y + 2*d3->velocity.y + d4->velocity.y);
        g->velocity.z  += sim->time_step/6.0 *
                          (d1->velocity.z + 2*d2->velocity.z + 2*d3->velocity.z + d4->velocity.z);
        g->magnetic.x  += sim->time_step/6.0 *
                          (d1->magnetic.x + 2*d2->magnetic.x + 2*d3->magnetic.x + d4->magnetic.x);
        g->magnetic.y  += sim->time_step/6.0 *
                          (d1->magnetic.y + 2*d2->magnetic.y + 2*d3->magnetic.y + d4->magnetic.y);
        g->magnetic.z  += sim->time_step/6.0 *
                          (d1->magnetic.z + 2*d2->magnetic.z + 2*d3->magnetic.z + d4->magnetic.z);
    }
}
