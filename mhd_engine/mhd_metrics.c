#include "mhd.h"


double mhd_get_simulation_time(MHDSimulation *sim) {
    if (!sim) return 0.0;
    return sim->current_time;
}

double mhd_get_energy_kinetic(MHDSimulation *sim) {
    if (!sim) return 0.0;
    // Oblicz aktualną energię kinetyczną
    sim->energy_kinetic = compute_energy_kinetic(sim);
    return sim->energy_kinetic;
}

double mhd_get_energy_magnetic(MHDSimulation *sim) {
    if (!sim) return 0.0;
    // Oblicz aktualną energię magnetyczną
    sim->energy_magnetic = compute_energy_magnetic(sim);
    return sim->energy_magnetic;
}

double mhd_get_energy_thermal(MHDSimulation *sim) {
    if (!sim) return 0.0;
    // Oblicz aktualną energię termiczną
    sim->energy_thermal = compute_energy_thermal(sim);
    return sim->energy_thermal;
}

double mhd_get_max_div_b(MHDSimulation *sim) {
    if (!sim) return 0.0;
    // Oblicz aktualną maksymalną dywergencję pola magnetycznego
    sim->max_div_b = compute_max_divergence_B(sim);
    return sim->max_div_b;
}
