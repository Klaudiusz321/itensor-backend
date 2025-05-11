/**
 * mhd_solver.c - Numerical solvers for MHD simulation
 * 
 * Implements Euler, Runge-Kutta 2nd order, and Runge-Kutta 4th order 
 * numerical methods for solving MHD equations.
 */

#include "mhd.h"
#include <math.h>

/**
 * Set the numerical method used for simulation
 */
void mhd_set_numerical_method(MHDSimulation *sim, NumericalMethod method) {
    if (!sim) return;
    sim->method = method;
}

/**
 * Simple divergence‐cleaning that operates on the derivative buffer
 */
static void apply_divergence_cleaning(GridCell *deriv, double div_B, double dt) {
    double clean_coeff = 0.1 * dt;
    deriv->magnetic.x -= clean_coeff * div_B;
    deriv->magnetic.y -= clean_coeff * div_B;
    deriv->magnetic.z -= clean_coeff * div_B;
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
    double eta   = sim->magnetic_viscosity;
    double sigma = sim->magnetic_conductivity;
    double cs2   = sim->sound_speed * sim->sound_speed;

    /* apply BC to the source buffer */
    mhd_apply_boundary_conditions(sim, source);

    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
            for (int k = 1; k < nz - 1; k++) {
                GridCell *c  = &source[i][j][k];
                GridCell *xp = &source[i+1][j][k];
                GridCell *xm = &source[i-1][j][k];
                GridCell *yp = &source[i][j+1][k];
                GridCell *ym = &source[i][j-1][k];
                GridCell *zp = &source[i][j][k+1];
                GridCell *zm = &source[i][j][k-1];

                // 1) divergence of v and B
                double dvx = (xp->velocity.x - xm->velocity.x)*0.5;
                double dvy = (yp->velocity.y - ym->velocity.y)*0.5;
                double dvz = (zp->velocity.z - zm->velocity.z)*0.5;
                double div_v = dvx + dvy + dvz;

                double dBx = (xp->magnetic.x - xm->magnetic.x)*0.5;
                double dBy = (yp->magnetic.y - ym->magnetic.y)*0.5;
                double dBz = (zp->magnetic.z - zm->magnetic.z)*0.5;
                double div_B = dBx + dBy + dBz;

                // 2) advection of v
                double drho_dx = (xp->density - xm->density)*0.5;
                double drho_dy = (yp->density - ym->density)*0.5;
                double drho_dz = (zp->density - zm->density)*0.5;
                double adv_rho = c->velocity.x*drho_dx
                               + c->velocity.y*drho_dy
                               + c->velocity.z*drho_dz;

                // 3) advection of v
                double vx_adv = c->velocity.x * dvx
                              + c->velocity.y * ((xp->velocity.x - xm->velocity.x)*0.5)
                              + c->velocity.z * ((zp->velocity.x - zm->velocity.x)*0.5);
                double vy_adv = c->velocity.x * ((xp->velocity.y - xm->velocity.y)*0.5)
                              + c->velocity.y * dvy
                              + c->velocity.z * ((zp->velocity.y - zm->velocity.y)*0.5);
                double vz_adv = c->velocity.x * ((xp->velocity.z - xm->velocity.z)*0.5)
                              + c->velocity.y * ((yp->velocity.z - ym->velocity.z)*0.5)
                              + c->velocity.z * dvz;

                // 4) curl(v×B)
                double curl_vB_x = ((yp->velocity.z * yp->magnetic.y)
                                   - (ym->velocity.z * ym->magnetic.y)) * 0.5
                                 - ((zp->velocity.y * zp->magnetic.z)
                                   - (zm->velocity.y * zm->magnetic.z)) * 0.5;
                double curl_vB_y = ((zp->velocity.x * zp->magnetic.z)
                                   - (zm->velocity.x * zm->magnetic.z)) * 0.5
                                 - ((xp->velocity.z * xp->magnetic.x)
                                   - (xm->velocity.z * xm->magnetic.x)) * 0.5;
                double curl_vB_z = ((xp->velocity.y * xp->magnetic.x)
                                   - (xm->velocity.y * xm->magnetic.x)) * 0.5
                                 - ((yp->velocity.x * yp->magnetic.y)
                                   - (ym->velocity.x * ym->magnetic.y)) * 0.5;

                // 5) Lorentz force
                double curl_B_x = (dBy - dBz);
                double curl_B_y = (dBz - dBx);
                double curl_B_z = (dBx - dBy);
                double lorentz_x = (curl_B_y * c->magnetic.z - curl_B_z * c->magnetic.y) * sigma;
                double lorentz_y = (curl_B_z * c->magnetic.x - curl_B_x * c->magnetic.z) * sigma;
                double lorentz_z = (curl_B_x * c->magnetic.y - curl_B_y * c->magnetic.x) * sigma;

                // 6) diffusion of B
                double lap_Bx = xp->magnetic.x + xm->magnetic.x
                              + yp->magnetic.x + ym->magnetic.x
                              + zp->magnetic.x + zm->magnetic.x
                              - 6.0*c->magnetic.x;
                double lap_By = xp->magnetic.y + xm->magnetic.y
                              + yp->magnetic.y + ym->magnetic.y
                              + zp->magnetic.y + zm->magnetic.y
                              - 6.0*c->magnetic.y;
                double lap_Bz = xp->magnetic.z + xm->magnetic.z
                              + yp->magnetic.z + ym->magnetic.z
                              + zp->magnetic.z + zm->magnetic.z
                              - 6.0*c->magnetic.z;
                double diff_Bx = eta * lap_Bx;
                double diff_By = eta * lap_By;
                double diff_Bz = eta * lap_Bz;

                // write derivatives
                GridCell *d = &derivatives[i][j][k];

                d->density     = -adv_rho - c->density * div_v;
                d->velocity.x  = -vx_adv
                               - (xp->pressure - xm->pressure)*0.5 / c->density
                               + lorentz_x / c->density;
                d->velocity.y  = -vy_adv
                               - (yp->pressure - ym->pressure)*0.5 / c->density
                               + lorentz_y / c->density;
                d->velocity.z  = -vz_adv
                               - (zp->pressure - zm->pressure)*0.5 / c->density
                               + lorentz_z / c->density;
                d->magnetic.x  = curl_vB_x + diff_Bx;
                d->magnetic.y  = curl_vB_y + diff_By;
                d->magnetic.z  = curl_vB_z + diff_Bz;
                d->pressure    = cs2 * d->density;
                d->temperature = d->pressure / c->density
                               - c->pressure * d->density / (c->density*c->density);

                // clean divergence
                apply_divergence_cleaning(d, div_B, sim->time_step);
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
        g->density    += sim->time_step * d->density;
        g->pressure   += sim->time_step * d->pressure;
        g->temperature+= sim->time_step * d->temperature;
        g->velocity.x += sim->time_step * d->velocity.x;
        g->velocity.y += sim->time_step * d->velocity.y;
        g->velocity.z += sim->time_step * d->velocity.z;
        g->magnetic.x += sim->time_step * d->magnetic.x;
        g->magnetic.y += sim->time_step * d->magnetic.y;
        g->magnetic.z += sim->time_step * d->magnetic.z;
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
        tmp.density     += 0.5*sim->time_step * d1->density;
        tmp.pressure    += 0.5*sim->time_step * d1->pressure;
        tmp.temperature += 0.5*sim->time_step * d1->temperature;
        tmp.velocity.x  += 0.5*sim->time_step * d1->velocity.x;
        tmp.velocity.y  += 0.5*sim->time_step * d1->velocity.y;
        tmp.velocity.z  += 0.5*sim->time_step * d1->velocity.z;
        tmp.magnetic.x  += 0.5*sim->time_step * d1->magnetic.x;
        tmp.magnetic.y  += 0.5*sim->time_step * d1->magnetic.y;
        tmp.magnetic.z  += 0.5*sim->time_step * d1->magnetic.z;
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
        g->density    += sim->time_step * d2->density;
        g->pressure   += sim->time_step * d2->pressure;
        g->temperature+= sim->time_step * d2->temperature;
        g->velocity.x += sim->time_step * d2->velocity.x;
        g->velocity.y += sim->time_step * d2->velocity.y;
        g->velocity.z += sim->time_step * d2->velocity.z;
        g->magnetic.x += sim->time_step * d2->magnetic.x;
        g->magnetic.y += sim->time_step * d2->magnetic.y;
        g->magnetic.z += sim->time_step * d2->magnetic.z;
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
