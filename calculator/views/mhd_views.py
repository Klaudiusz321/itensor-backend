import math
import traceback

from rest_framework.views import APIView
from rest_framework.response import Response

from calculator.views.mhd.utils.mhd_interface import MHDInterface, MHDLibraryError


class MHDSimulationView(APIView):
    """
    API endpoint to run MHD simulations and return field data and metrics as JSON.
    """
    def post(self, request):
        try:
            data = request.data

            # 1) Grid dimensions
            try:
                gx = int(data["grid_x"])
                gy = int(data["grid_y"])
                gz = int(data["grid_z"])
                if gx <= 0 or gy <= 0 or gz <= 0:
                    raise ValueError
            except Exception:
                return Response(
                    {"status": "error",
                     "error": "grid_x, grid_y and grid_z must be positive integers"},
                    status=400
                )

            # 2) Time parameters
            duration  = float(data.get("duration", 5.0))
            time_step = float(data.get("time_step", 0.01))

            # 3) Numerical method
            method = int(data.get("method", 0))

            # 4) Fluid properties
            density     = float(data.get("initial_density", 1.0))
            pressure    = float(data.get("initial_pressure", 1.0))
            temperature = float(data.get("initial_temperature", 300.0))

            # 5) Initial velocity
            iv = data.get("initial_velocity", [0.0, 0.0, 0.0])
            try:
                vx, vy, vz = map(float, iv)
            except Exception:
                return Response(
                    {"status": "error",
                     "error": "initial_velocity must be an array of three floats"},
                    status=400
                )

            # 6) Initial magnetic field
            imf = data.get("initial_magnetic_field", [0.0, 0.0, 0.0])
            try:
                bx, by, bz = map(float, imf)
            except Exception:
                return Response(
                    {"status": "error",
                     "error": "initial_magnetic_field must be an array of three floats"},
                    status=400
                )

            # 7) Boundary, conductivity, viscosity, disturbances...
            boundary_type = int(data.get("boundary_type", 0))
            conductivity  = float(data.get("magnetic_conductivity", 0.0))
            viscosity     = float(data.get("magnetic_viscosity", 0.0))

            # dynamic field
            dfc = data.get("dynamic_field_changes", {})
            dfc_enabled = bool(dfc.get("enabled", False))
            dfc_type    = int(dfc.get("type", 0))
            dfc_amp     = float(dfc.get("amplitude", 0.0))
            dfc_freq    = float(dfc.get("frequency", 0.0))

            # turbulence
            turb_enabled  = bool(data.get("turbulence_enabled", False))
            turb_strength = float(data.get("turbulence_strength", 0.0))

            # disturbances
            ds = data.get("disturbance_settings", {})
            ds_enabled   = bool(ds.get("enabled", False))
            ds_freq      = float(ds.get("frequency", 0.0))
            ds_magnitude = float(ds.get("magnitude", 0.0))

            # random disturbances
            rd = data.get("random_disturbances", {})
            rd_enabled   = bool(rd.get("enabled", False))
            rd_magnitude = float(rd.get("magnitude", 0.0))

            # particle interaction
            particle_enabled = bool(data.get("particle_interaction_enabled", False))

            # spatial gradients
            sg = data.get("spatial_gradients", {})

            results = {}
            metrics = {}

            # --- RUN MHD SIMULATION ---
            with MHDInterface(gx, gy, gz) as engine:
                lib = engine._lib
                sim = engine._sim_handle

                # time & method
                lib.mhd_set_time_step(sim, time_step)
                lib.mhd_set_total_time(sim, duration)
                lib.mhd_set_numerical_method(sim, method)

                # fluid init
                lib.mhd_initialize_fluid_parameters(sim, density, pressure, temperature)
                lib.mhd_set_initial_velocity(sim, vx, vy, vz)
                lib.mhd_initialize_magnetic_field(sim, bx, by, bz)
                lib.mhd_set_boundary_type(sim, boundary_type)

                # magnetic props
                lib.mhd_set_magnetic_conductivity(sim, conductivity)
                lib.mhd_set_magnetic_viscosity(sim, viscosity)

                # dynamic field
                lib.mhd_enable_dynamic_field(sim, dfc_enabled)
                if dfc_enabled:
                    lib.mhd_set_dynamic_field_type(sim, dfc_type)
                    lib.mhd_set_dynamic_field_amplitude(sim, dfc_amp)
                    lib.mhd_set_dynamic_field_frequency(sim, dfc_freq)

                # turbulence
                lib.mhd_enable_turbulence(sim, turb_enabled)
                if turb_enabled:
                    lib.mhd_set_turbulence_strength(sim, turb_strength)

                # disturbances
                lib.mhd_enable_disturbance(sim, ds_enabled)
                if ds_enabled:
                    lib.mhd_set_disturbance_frequency(sim, ds_freq)
                    lib.mhd_set_disturbance_magnitude(sim, ds_magnitude)

                # random disturbances
                lib.mhd_enable_random_disturbances(sim, rd_enabled)
                if rd_enabled:
                    lib.mhd_set_random_disturbance_magnitude(sim, rd_magnitude)

                # particles & spatial gradients
                lib.mhd_enable_particle_interaction(sim, particle_enabled)
                for field_name, enabled in sg.items():
                    lib.mhd_enable_spatial_gradient(sim, field_name.encode('utf-8'), int(enabled))

                # run
                lib.mhd_run_simulation(sim)

                # 1) EXPORT SUROWE POLA
                for field in (
                    "density",
                    "temperature",
                    "pressure",
                    "velocity_field",
                    "magnetic_vector",
                ):
                    flat = engine.export_field(field)
                    # skalarne pola:
                    if field in ("density", "temperature", "pressure"):
                        flat = [row[3] for row in flat]
                    # wektorowe:
                    else:
                        flat = [row[3:6] for row in flat]

                    # 2D vs 3D:
                    if gz == 1:
                        grid = [flat[i * gy:(i + 1) * gy] for i in range(gx)]
                    else:
                        grid = [
                            [
                                [flat[(x * gy + y) * gz + z] for z in range(gz)]
                                for y in range(gy)
                            ]
                            for x in range(gx)
                        ]

                    results[field] = grid

                # ↑ po tej pętli mamy w results:
                #   "density", "temperature", "pressure",
                #   "velocity_field" (wektory), "magnetic_vector" (wektory)

                # 2) DODAJEMY POLA MAGNITUD
                # velocity magnitude
                vel = results["velocity_field"]
                results["velocity_magnitude"] = [
                    [math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in row]
                    for row in vel
                ]

                mag = results["magnetic_vector"]
                results["magnetic_field_magnitude"] = [
                    [math.sqrt(m[0]**2 + m[1]**2 + m[2]**2) for m in row]
                    for row in mag
                ]

                # 2) aliasy na nazwy, których front-end używa w vizField:
                results["velocity"] = results["velocity_magnitude"]
                results["magnetic_magnitude"] = results["magnetic_field_magnitude"]

                # 3) wektor magnetyczny pod kluczem 'magnetic'
                results["magnetic"] = results.pop("magnetic_vector")

                # 3) EXPORT METRYK
                metrics["time"]            = lib.mhd_get_simulation_time(sim)
                metrics["energy_kinetic"]  = lib.mhd_get_energy_kinetic(sim)
                metrics["energy_magnetic"] = lib.mhd_get_energy_magnetic(sim)
                metrics["energy_thermal"]  = lib.mhd_get_energy_thermal(sim)
                metrics["max_div_b"]       = lib.mhd_get_max_div_b(sim)

            # KONIEC with…

            # 4) ZWRÓĆ WSZYSTKO DO FRONTENDU
            return Response({
                "status":  "ok",
                "results": results,
                "metrics": metrics
            })

        except MHDLibraryError as e:
            return Response({"status": "error", "error": str(e)}, status=500)

        except Exception as e:
            traceback.print_exc()
            return Response({"status": "error", "error": str(e)}, status=500)
