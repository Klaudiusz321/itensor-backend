from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings
import os
import csv
import traceback
import math
from calculator.views.mhd.utils.mhd_interface import lib

class MHDSimulationView(APIView):
    def post(self, request):
        try:
            data = request.data
            if all(k in data for k in ("grid_x","grid_y","grid_z")):
                gx,gy,gz = map(int,(data["grid_x"],data["grid_y"],data["grid_z"]))
            elif "grid_resolution" in data:
                res = data["grid_resolution"]
                if len(res)==2:
                    gx,gy = map(int,res); gz=1
                elif len(res)==3:
                    gx,gy,gz = map(int,res)
                else:
                    return Response({"status":"error","error":"Invalid grid_resolution"},status=400)
            else:
                return Response({"status":"error","error":"Missing grid parameters"},status=400)

            os.chdir(settings.BASE_DIR)
            sim = lib.mhd_initialize(gx,gy,gz)

            lib.mhd_set_time_step(sim,float(data.get("time_step",0.01)))
            lib.mhd_set_total_time(sim,float(data.get("total_time",1.0)))
            lib.mhd_set_numerical_method(sim,int(data.get("method",1)))

            lib.mhd_initialize_fluid_parameters(
                sim,
                float(data.get("initial_density",1.0)),
                float(data.get("initial_pressure",1.0)),
                float(data.get("initial_temperature",1.0))
            )
            vx,vy,vz = data.get("initial_velocity",[0,0,0])
            lib.mhd_set_initial_velocity(sim,float(vx),float(vy),float(vz))

            bx,by,bz = data.get("initial_magnetic_field",[0,0,1])
            lib.mhd_initialize_magnetic_field(sim,float(bx),float(by),float(bz))

            lib.mhd_set_boundary_type(sim,int(data.get("boundary_type",3)))

            if float(data.get("turbulence_intensity",0))>0:
                lib.mhd_apply_turbulence(sim,float(data["turbulence_intensity"]))
            if data.get("use_spatial_gradients",False):
                lib.mhd_apply_spatial_gradients(sim,True)

            lib.mhd_run_simulation(sim)

            exports = {
                "density":"density.csv",
                "pressure":"pressure.csv",
                "temperature":"temperature.csv",
                "velocity_x":"velocity_x.csv",
                "velocity_y":"velocity_y.csv",
                "magnetic_magnitude":"magnetic.csv"
            }
            generated = []
            for field,fname in exports.items():
                try:
                    lib.mhd_export_field_data(
                        sim,
                        field.encode("ascii"),
                        fname.encode("ascii")
                    )
                    generated.append(field)
                except AttributeError:
                    pass

            lib.mhd_free(sim)

            def load_csv(fname):
                path = os.path.join(settings.BASE_DIR, fname)
                rows = []
                with open(path, newline="") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if not row: continue
                        clean = []
                        for x in row:
                            try:
                                v = float(x)
                                clean.append(v if math.isfinite(v) else None)
                            except:
                                clean.append(None)
                        rows.append(clean)
                return rows

            results = {}
            if "density" in generated:
                results["density"] = load_csv("density.csv")
            if "pressure" in generated:
                results["pressure"] = load_csv("pressure.csv")
            if "temperature" in generated:
                results["temperature"] = load_csv("temperature.csv")
            if "velocity_x" in generated and "velocity_y" in generated:
                results["velocity"] = [
                    load_csv("velocity_x.csv"),
                    load_csv("velocity_y.csv")
                ]
            if "magnetic_magnitude" in generated:
                results["magnetic"] = load_csv("magnetic.csv")

            return Response({"status":"ok","results":results})
        except Exception as e:
            traceback.print_exc()
            return Response({"status":"error","error":str(e)},status=500)
