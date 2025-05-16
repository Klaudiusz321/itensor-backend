import os
import sys
import threading
import ctypes
import logging
import tempfile

from django.conf import settings

logger = logging.getLogger(__name__)

# Dozwolone pola do eksportu
ALLOWED_FIELDS = {
    'density', 'velocity', 'magnetic_field', 'temperature',
    'pressure', 'velocity_field', 'magnetic_vector'
}


class MHDLibraryError(Exception):
    pass


class MHDSimulation(ctypes.Structure):
    pass


class MHDInterface:
    _lib = None
    _initialized = False
    _init_lock = threading.Lock()

    def __init__(self, nx=100, ny=100, nz=100):
        # Jednorazowe ładowanie i rejestracja prototypów
        with MHDInterface._init_lock:
            if not MHDInterface._initialized:
                MHDInterface._load_library()
                MHDInterface._register_functions()
                MHDInterface._initialized = True

        self.nx, self.ny, self.nz = nx, ny, nz
        self._call_lock = threading.Lock()
        self._sim_handle = None

    @classmethod
    def _load_library(cls):
        # Wybór nazwy biblioteki zależnie od platformy
        lib_name = 'libmhd.dll' if os.name == 'nt' else 'libmhd.so'
        lib_path = os.path.join(settings.BASE_DIR, 'mhd_engine', lib_name)

        if not os.path.isfile(lib_path):
            raise MHDLibraryError(f"MHD library not found at {lib_path}")

        print(f"[MHDInterface] ▶ Loading MHD engine from: {lib_path}", file=sys.stderr)
        try:
            cls._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise MHDLibraryError(f"Failed to load MHD library: {e}")
        print(f"[MHDInterface] ✔ Loaded: {cls._lib._name}", file=sys.stderr)

    @classmethod
    def _register_functions(cls):
        lib = cls._lib
        
        
        # --- Simulation init/free ---
        lib.mhd_initialize.restype = ctypes.POINTER(MHDSimulation)
        lib.mhd_initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.mhd_free.restype = None
        lib.mhd_free.argtypes = [ctypes.POINTER(MHDSimulation)]
        lib.mhd_initialize_simulation = lib.mhd_initialize
        lib.mhd_free_simulation = lib.mhd_free

        # --- Export field data ---
        lib.mhd_export_field_data.restype = None
        lib.mhd_export_field_data.argtypes = [
            ctypes.POINTER(MHDSimulation),
            ctypes.c_char_p,
            ctypes.c_char_p
        ]

        # --- Time parameters ---
        lib.mhd_set_time_step.restype = None
        lib.mhd_set_time_step.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_double]
        lib.mhd_set_total_time.restype = None
        lib.mhd_set_total_time.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_double]

        # --- Fluid parameters ---
        lib.mhd_initialize_fluid_parameters.restype = None
        lib.mhd_initialize_fluid_parameters.argtypes = [
            ctypes.POINTER(MHDSimulation), ctypes.c_double, ctypes.c_double, ctypes.c_double
        ]

        # --- Magnetic field init ---
        lib.mhd_initialize_magnetic_field.restype = None
        lib.mhd_initialize_magnetic_field.argtypes = [
            ctypes.POINTER(MHDSimulation), ctypes.c_double, ctypes.c_double, ctypes.c_double
        ]

        # --- Initial velocity ---
        lib.mhd_set_initial_velocity.restype = None
        lib.mhd_set_initial_velocity.argtypes = [
            ctypes.POINTER(MHDSimulation), ctypes.c_double, ctypes.c_double, ctypes.c_double
        ]

        # --- Boundary & method ---
        lib.mhd_set_boundary_type.restype = None
        lib.mhd_set_boundary_type.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_int]
        lib.mhd_set_numerical_method.restype = None
        lib.mhd_set_numerical_method.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_int]

        # --- Run simulation ---
        lib.mhd_run_simulation.restype = None
        lib.mhd_run_simulation.argtypes = [ctypes.POINTER(MHDSimulation)]

        # --- Extended setters ---
        lib.mhd_set_magnetic_conductivity.restype = None
        lib.mhd_set_magnetic_conductivity.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_double]
        lib.mhd_set_magnetic_viscosity.restype = None
        lib.mhd_set_magnetic_viscosity.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_double]


        if hasattr(lib, 'mhd_enable_dynamic_field'):
            lib.mhd_enable_dynamic_field.restype = None
            lib.mhd_enable_dynamic_field.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_int]
        else:
            logger.warning("Function 'mhd_enable_dynamic_field' not available")

        if hasattr(lib, 'mhd_set_dynamic_field_type'):
            lib.mhd_set_dynamic_field_type.restype = None
            lib.mhd_set_dynamic_field_type.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_int]
        else:
            logger.warning("Function 'mhd_set_dynamic_field_type' not available")
        lib.mhd_set_dynamic_field_amplitude.restype = None
        lib.mhd_set_dynamic_field_amplitude.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_double]
        lib.mhd_set_dynamic_field_frequency.restype = None
        lib.mhd_set_dynamic_field_frequency.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_double]

        lib.mhd_enable_turbulence.restype = None
        lib.mhd_enable_turbulence.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_int]
        lib.mhd_set_turbulence_strength.restype = None
        lib.mhd_set_turbulence_strength.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_double]

        lib.mhd_enable_disturbance.restype = None
        lib.mhd_enable_disturbance.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_int]
        lib.mhd_set_disturbance_frequency.restype = None
        lib.mhd_set_disturbance_frequency.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_double]
        lib.mhd_set_disturbance_magnitude.restype = None
        lib.mhd_set_disturbance_magnitude.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_double]

        lib.mhd_enable_random_disturbances.restype = None
        lib.mhd_enable_random_disturbances.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_int]
        lib.mhd_set_random_disturbance_magnitude.restype = None
        lib.mhd_set_random_disturbance_magnitude.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_double]

        lib.mhd_enable_particle_interaction.restype = None
        lib.mhd_enable_particle_interaction.argtypes = [ctypes.POINTER(MHDSimulation), ctypes.c_int]

        lib.mhd_enable_spatial_gradient.restype = None
        lib.mhd_enable_spatial_gradient.argtypes = [
            ctypes.POINTER(MHDSimulation), ctypes.c_char_p, ctypes.c_int
        ]

        # --- Getters ---
        lib.mhd_get_simulation_time.restype = ctypes.c_double
        lib.mhd_get_simulation_time.argtypes = [ctypes.POINTER(MHDSimulation)]
        lib.mhd_get_energy_kinetic.restype = ctypes.c_double
        lib.mhd_get_energy_kinetic.argtypes = [ctypes.POINTER(MHDSimulation)]
        lib.mhd_get_energy_magnetic.restype = ctypes.c_double
        lib.mhd_get_energy_magnetic.argtypes = [ctypes.POINTER(MHDSimulation)]
        lib.mhd_get_energy_thermal.restype = ctypes.c_double
        lib.mhd_get_energy_thermal.argtypes = [ctypes.POINTER(MHDSimulation)]
        lib.mhd_get_max_div_b.restype = ctypes.c_double
        lib.mhd_get_max_div_b.argtypes = [ctypes.POINTER(MHDSimulation)]

    def __enter__(self):
        logger.debug(f"[MHDInterface] ▶ init({self.nx},{self.ny},{self.nz})")
        self._sim_handle = self._lib.mhd_initialize_simulation(self.nx, self.ny, self.nz)
        if not self._sim_handle:
            raise MHDLibraryError("mhd_initialize returned NULL")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._sim_handle:
            self._lib.mhd_free_simulation(self._sim_handle)
        self._sim_handle = None

    # Core setters
    def set_time_step(self, dt: float):
        with self._call_lock:
            self._lib.mhd_set_time_step(self._sim_handle, dt)

    def set_total_time(self, total: float):
        with self._call_lock:
            self._lib.mhd_set_total_time(self._sim_handle, total)

    def set_fluid_parameters(self, density: float, pressure: float, temperature: float):
        with self._call_lock:
            self._lib.mhd_initialize_fluid_parameters(self._sim_handle, density, pressure, temperature)

    def set_magnetic_field(self, bx: float, by: float, bz: float):
        with self._call_lock:
            self._lib.mhd_initialize_magnetic_field(self._sim_handle, bx, by, bz)

    def set_initial_velocity(self, vx: float, vy: float, vz: float):
        with self._call_lock:
            self._lib.mhd_set_initial_velocity(self._sim_handle, vx, vy, vz)

    def set_boundary_type(self, btype: int):
        with self._call_lock:
            self._lib.mhd_set_boundary_type(self._sim_handle, btype)

    def set_numerical_method(self, method: int):
        with self._call_lock:
            self._lib.mhd_set_numerical_method(self._sim_handle, method)

    def run(self):
        with self._call_lock:
            self._lib.mhd_run_simulation(self._sim_handle)

    # Extended wrappers
    def set_magnetic_conductivity(self, cond: float):
        with self._call_lock:
            self._lib.mhd_set_magnetic_conductivity(self._sim_handle, cond)

    def set_magnetic_viscosity(self, vis: float):
        with self._call_lock:
            self._lib.mhd_set_magnetic_viscosity(self._sim_handle, vis)

    def enable_dynamic_field(self, enabled: bool):
        fn = getattr(self._lib, 'mhd_enable_dynamic_field', None)
        if fn is None:
            with self._call_lock:
                self._lib.mhd_enable_dynamic_field(self._sim_handle, int(enabled))
        else:
            logger.debug("Function 'mhd_enable_dynamic_field' not available")

    def set_dynamic_field_type(self, t: int):
        with self._call_lock:
            self._lib.mhd_set_dynamic_field_type(self._sim_handle, t)

    def set_dynamic_field_amplitude(self, amp: float):
        with self._call_lock:
            self._lib.mhd_set_dynamic_field_amplitude(self._sim_handle, amp)

    def set_dynamic_field_frequency(self, freq: float):
        with self._call_lock:
            self._lib.mhd_set_dynamic_field_frequency(self._sim_handle, freq)

    def enable_turbulence(self, enabled: bool):
        with self._call_lock:
            self._lib.mhd_enable_turbulence(self._sim_handle, int(enabled))

    def set_turbulence_strength(self, strength: float):
        with self._call_lock:
            self._lib.mhd_set_turbulence_strength(self._sim_handle, strength)

    def enable_disturbance(self, enabled: bool):
        with self._call_lock:
            self._lib.mhd_enable_disturbance(self._sim_handle, int(enabled))

    def set_disturbance_frequency(self, freq: float):
        with self._call_lock:
            self._lib.mhd_set_disturbance_frequency(self._sim_handle, freq)

    def set_disturbance_magnitude(self, mag: float):
        with self._call_lock:
            self._lib.mhd_set_disturbance_magnitude(self._sim_handle, mag)

    def enable_random_disturbances(self, enabled: bool):
        with self._call_lock:
            self._lib.mhd_enable_random_disturbances(self._sim_handle, int(enabled))

    def set_random_disturbance_magnitude(self, mag: float):
        with self._call_lock:
            self._lib.mhd_set_random_disturbance_magnitude(self._sim_handle, mag)

    def enable_particle_interaction(self, enabled: bool):
        with self._call_lock:
            self._lib.mhd_enable_particle_interaction(self._sim_handle, int(enabled))

    def enable_spatial_gradient(self, field_name: str, enabled: bool):
        with self._call_lock:
            self._lib.mhd_enable_spatial_gradient(
                self._sim_handle,
                field_name.encode('utf-8'),
                int(enabled)
            )

    def export_field(self, field_name: str):
        """
        Eksportuje pole symulacji do Pythonowej listy, ignorując
        wszystkie linie, których nie da się przekonwertować na float.
        """
        if field_name not in ALLOWED_FIELDS:
            raise ValueError(f"Invalid field: {field_name!r}")

        # Tworzymy plik tymczasowy
        tmp = tempfile.NamedTemporaryFile(prefix='mhd_', suffix='.csv', delete=False)
        tmp.close()
        fname = tmp.name

        # Wywołujemy C-ową funkcję eksportu
        with self._call_lock:
            self._lib.mhd_export_field_data(
                self._sim_handle,
                field_name.encode('utf-8'),
                fname.encode('utf-8')
            )

        data = []
        # Czytamy i filtrujemy
        with open(fname, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                try:
                    data.append([float(p) for p in parts])
                except ValueError:
                    continue

        os.remove(fname)
        return data