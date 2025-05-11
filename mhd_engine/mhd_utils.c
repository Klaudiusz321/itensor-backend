/**
 * mhd_utils.c – Utility functions for MHD simulation
 */

#include "mhd.h"
#include <stdlib.h>   // malloc, calloc, free, exit
#include <stdio.h>    // fprintf, stderr
#include <time.h>     // time, srand

/**
 * Helper macro do sprzątania przy częściowej alokacji:
 *   gridptr  – wskaźnik na tablicę [X][Y][...]
 *   X, Y     – wymiary pierwszych dwóch poziomów
 */
#define FREE_PARTIAL(gridptr, X, Y)                 \
    do {                                            \
        if (gridptr) {                              \
            for (size_t _i = 0; _i < (X); ++_i) {   \
                if ((gridptr)[_i]) {                \
                    for (size_t _j = 0; _j < (Y); ++_j) \
                        free((gridptr)[_i][_j]);     \
                    free((gridptr)[_i]);             \
                }                                   \
            }                                       \
            free(gridptr);                          \
            gridptr = NULL;                         \
        }                                           \
    } while (0)


/**
 * Allocate both the main grid and temporary grid.
 * On any allocation failure, frees what było już zaalokowane i kończy program.
 */
void mhd_allocate_grids(MHDSimulation *sim) {
    if (!sim) return;

    size_t nx = sim->grid_size_x;
    size_t ny = sim->grid_size_y;
    size_t nz = sim->grid_size_z;

    // --- Alokacja głównej siatki
    sim->grid = malloc(nx * sizeof(GridCell**));
    if (!sim->grid) {
        fprintf(stderr, "Error: malloc(grid) failed\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < nx; ++i) {
        sim->grid[i] = malloc(ny * sizeof(GridCell*));
        if (!sim->grid[i]) {
            fprintf(stderr, "Error: malloc(grid[%zu]) failed\n", i);
            FREE_PARTIAL(sim->grid, i, ny);
            exit(EXIT_FAILURE);
        }
        for (size_t j = 0; j < ny; ++j) {
            sim->grid[i][j] = calloc(nz, sizeof(GridCell));
            if (!sim->grid[i][j]) {
                fprintf(stderr, "Error: calloc(grid[%zu][%zu]) failed\n", i, j);
                // sprzątnij tę kolumnę i całą dotychczasową część
                for (size_t jj = 0; jj < j; ++jj)
                    free(sim->grid[i][jj]);
                free(sim->grid[i]);
                FREE_PARTIAL(sim->grid, i, ny);
                exit(EXIT_FAILURE);
            }
        }
    }

    // --- Alokacja pomocniczej siatki temp_grid
    sim->temp_grid = malloc(nx * sizeof(GridCell**));
    if (!sim->temp_grid) {
        fprintf(stderr, "Error: malloc(temp_grid) failed\n");
        FREE_PARTIAL(sim->grid, nx, ny);
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < nx; ++i) {
        sim->temp_grid[i] = malloc(ny * sizeof(GridCell*));
        if (!sim->temp_grid[i]) {
            fprintf(stderr, "Error: malloc(temp_grid[%zu]) failed\n", i);
            FREE_PARTIAL(sim->temp_grid, i, ny);
            FREE_PARTIAL(sim->grid,    nx, ny);
            exit(EXIT_FAILURE);
        }
        for (size_t j = 0; j < ny; ++j) {
            sim->temp_grid[i][j] = calloc(nz, sizeof(GridCell));
            if (!sim->temp_grid[i][j]) {
                fprintf(stderr, "Error: calloc(temp_grid[%zu][%zu]) failed\n", i, j);
                for (size_t jj = 0; jj < j; ++jj)
                    free(sim->temp_grid[i][jj]);
                free(sim->temp_grid[i]);
                FREE_PARTIAL(sim->temp_grid, i, ny);
                FREE_PARTIAL(sim->grid,    nx, ny);
                exit(EXIT_FAILURE);
            }
        }
    }
}

/**
 * Free both the main grid and the temporary grid.
 */
void mhd_free_grids(MHDSimulation *sim) {
    if (!sim) return;

    size_t nx = sim->grid_size_x;
    size_t ny = sim->grid_size_y;

    // Zwolnij grid
    if (sim->grid) {
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j)
                free(sim->grid[i][j]);
            free(sim->grid[i]);
        }
        free(sim->grid);
        sim->grid = NULL;
    }

    // Zwolnij temp_grid
    if (sim->temp_grid) {
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j)
                free(sim->temp_grid[i][j]);
            free(sim->temp_grid[i]);
        }
        free(sim->temp_grid);
        sim->temp_grid = NULL;
    }
}

/**
 * Seed RNG raz na początku symulacji.
 * Wywołaj np. w mhd_initialize() tuż po alokacji.
 */
void mhd_seed_rng(void) {
    srand((unsigned int)time(NULL));
}

#undef FREE_PARTIAL
