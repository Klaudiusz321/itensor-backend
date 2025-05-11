import ctypes
import os
from django.conf import settings
LIB_PATH = os.path.join(settings.BASE_DIR, "mhd_engine", "libmhd.so")
lib = ctypes.CDLL(LIB_PATH)

# „Opaque” struktura – traktujemy ją jako wskaźnik
class MHDSimulation(ctypes.Structure):
    pass

# Sygnatury funkcji z C
lib.mhd_initialize.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
lib.mhd_initialize.restype  = ctypes.POINTER(MHDSimulation)

lib.mhd_set_time_step.argtypes  = (ctypes.POINTER(MHDSimulation), ctypes.c_double)
lib.mhd_set_total_time.argtypes = (ctypes.POINTER(MHDSimulation), ctypes.c_double)
lib.mhd_set_numerical_method.argtypes = (ctypes.POINTER(MHDSimulation), ctypes.c_int)

lib.mhd_initialize_fluid_parameters.argtypes = (
    ctypes.POINTER(MHDSimulation),
    ctypes.c_double, ctypes.c_double, ctypes.c_double
)
lib.mhd_set_initial_velocity.argtypes = (
    ctypes.POINTER(MHDSimulation),
    ctypes.c_double, ctypes.c_double, ctypes.c_double
)

lib.mhd_initialize_magnetic_field.argtypes = (
    ctypes.POINTER(MHDSimulation),
    ctypes.c_double, ctypes.c_double, ctypes.c_double
)
lib.mhd_set_boundary_type.argtypes = (ctypes.POINTER(MHDSimulation), ctypes.c_int)

lib.mhd_apply_turbulence.argtypes        = (ctypes.POINTER(MHDSimulation), ctypes.c_double)
lib.mhd_apply_spatial_gradients.argtypes = (ctypes.POINTER(MHDSimulation), ctypes.c_bool)

lib.mhd_run_simulation.argtypes = (ctypes.POINTER(MHDSimulation),)
lib.mhd_free.argtypes           = (ctypes.POINTER(MHDSimulation),)

# Jeśli potrzebujesz też wrapperów dla exportu, możesz je dorzucić tu.
# Na przykład, by zwrócić ścieżkę pliku wygenerowanego po stronie C:

def export_csv(field_name: str, filename: str) -> str:
    return os.path.join( filename)
