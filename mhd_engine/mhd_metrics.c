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

/**
 * Update all simulation metrics at once
 * This function calculates and updates all energy metrics and magnetic field divergence
 * for efficient reporting to the frontend
 * 
 * @param sim Pointer to the simulation context
 */
void mhd_update_all_metrics(MHDSimulation *sim) {
    if (!sim) return;
    
    // Update all metrics
    sim->energy_kinetic = compute_energy_kinetic(sim);
    sim->energy_magnetic = compute_energy_magnetic(sim);
    sim->energy_thermal = compute_energy_thermal(sim);
    sim->max_div_b = compute_max_divergence_B(sim);
    
    // Log the metrics for debugging
    printf("Updated metrics: E_kin=%.3f, E_mag=%.3f, E_th=%.3f, max_div_B=%.6f\n",
           sim->energy_kinetic, sim->energy_magnetic, sim->energy_thermal, sim->max_div_b);
}
