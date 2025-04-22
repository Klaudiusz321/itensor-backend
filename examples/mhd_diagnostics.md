# MHD Diagnostic Tools

This document provides information on how to use the diagnostic tools for MHD (Magnetohydrodynamics) simulations in the Tensor-backend-calculator.

## Available Diagnostic Endpoints

The following diagnostic endpoints are available:

- `/api/mhd-diagnostic/` - Basic diagnostics for MHD simulations
- `/api/mhd-stress-test/` - Stress test for identifying parameter combinations that cause instability

## Basic Diagnostic

The basic diagnostic endpoint checks:
- Initialization of the MHD system
- Buffer allocation and memory usage
- Single time step execution
- Memory diagnostics

### Example Request

```json
POST /api/mhd-diagnostic/
{
  "simulation_type": "magnetic_rotor",
  "resolution": [32, 32],
  "domain_size": [[0,1],[0,1]],
  "gamma": 1.67
}
```

### Example Response

```json
{
  "import_success": true,
  "initialization": {
    "success": true,
    "system_info": {
      "resolution": [32, 32],
      "coordinate_system": "{'name': 'cartesian', 'coordinates': ['x', 'y'], 'transformation': None}",
      "has_buffers": true,
      "buffer_keys": ["temp_density", "temp_momentum", "temp_energy", "temp_magnetic", "density_flux", "momentum_flux", "energy_flux", "magnetic_flux"],
      "dt": 0.004545454545454545,
      "cfl": 0.4
    },
    "initial_check": {
      "density": {
        "has_nan": false,
        "has_inf": false,
        "min": 1.0,
        "max": 10.0
      },
      "pressure": {
        "has_nan": false,
        "has_inf": false,
        "min": 1.0,
        "max": 1.0
      }
    }
  },
  "buffer_check": {
    "success": true,
    "buffer_sizes": {
      "temp_density": [32, 32],
      "temp_momentum": [2, 32, 32],
      "temp_energy": [32, 32],
      "temp_magnetic": [2, 32, 32],
      "density_flux": [32, 32],
      "momentum_flux": [2, 32, 32],
      "energy_flux": [32, 32],
      "magnetic_flux": [2, 32, 32]
    }
  },
  "single_step": {
    "success": true,
    "method": "safe_advance",
    "dt": 0.0018181818181818181,
    "post_step_check": {
      "density": {
        "has_nan": false,
        "has_inf": false,
        "min": 0.9959603706212122,
        "max": 10.002227384417677
      },
      "pressure": {
        "has_nan": false,
        "has_inf": false,
        "min": 0.9931853694755834,
        "max": 1.0056248807873713
      }
    }
  },
  "memory_check": {
    "success": true
  },
  "memory_diagnostics": {
    "system_info": {
      "dimension": 2,
      "shape": [32, 32],
      "resolution": [32, 32],
      "coordinate_system": "{'name': 'cartesian', 'coordinates': ['x', 'y'], 'transformation': None}",
      "time": 0.0018181818181818181,
      "cfl": 0.4
    },
    "memory_usage": {
      "density": {
        "size_bytes": 8192,
        "size_mb": 0.0078125,
        "shape": [32, 32],
        "dtype": "float64"
      },
      "velocity_0": {
        "size_bytes": 8192,
        "size_mb": 0.0078125,
        "shape": [32, 32],
        "dtype": "float64"
      },
      "velocity_1": {
        "size_bytes": 8192,
        "size_mb": 0.0078125,
        "shape": [32, 32],
        "dtype": "float64"
      },
      "pressure": {
        "size_bytes": 8192,
        "size_mb": 0.0078125,
        "shape": [32, 32],
        "dtype": "float64"
      },
      "magnetic_0": {
        "size_bytes": 8192,
        "size_mb": 0.0078125,
        "shape": [32, 32],
        "dtype": "float64"
      },
      "magnetic_1": {
        "size_bytes": 8192,
        "size_mb": 0.0078125,
        "shape": [32, 32],
        "dtype": "float64"
      },
      "total": {
        "size_bytes": 311296,
        "size_mb": 0.296875,
        "breakdown": {
          "density": 8192,
          "velocity": 16384,
          "pressure": 8192,
          "magnetic": 16384,
          "buffer": 131072,
          "conserved": 131072
        }
      }
    },
    "buffer_info": {
      "temp_density": {
        "size_bytes": 8192,
        "size_mb": 0.0078125,
        "shape": [32, 32],
        "dtype": "float64"
      },
      "temp_momentum": {
        "size_bytes": 16384,
        "size_mb": 0.015625,
        "shape": [2, 32, 32],
        "dtype": "float64"
      },
      // More buffer info...
    },
    "state_consistency": {
      "grid_dimensions_match": true,
      "density_contains_nan_or_inf": false,
      "pressure_contains_nan_or_inf": false,
      "has_nan_or_inf": false,
      "max_div_b": 1.2345678901234567e-15,
      "divergence_free": true,
      "max_velocity": 2.0,
      "velocity_reasonable": true
    },
    "memory_management": {
      "gc_objects_before": 12345,
      "gc_objects_after": 12346,
      "difference": 1,
      "memory_leak_likely": false
    }
  },
  "versions": {
    "numpy": "1.24.3",
    "numba": "0.57.1"
  }
}
```

## Stress Test

The stress test endpoint runs multiple simulations with different parameter combinations to identify which configurations cause instability.

### Example Request

```json
POST /api/mhd-stress-test/
{
  "simulation_type": "orszag_tang_vortex",
  "base_resolution": [48, 48],
  "gamma_values": [1.4, 1.67, 2.0],
  "cfl_values": [0.2, 0.4, 0.6, 0.8],
  "test_duration": 0.1
}
```

### Example Response

```json
{
  "simulation_type": "orszag_tang_vortex",
  "base_resolution": [48, 48],
  "gamma_values": [1.4, 1.67, 2.0],
  "cfl_values": [0.2, 0.4, 0.6, 0.8],
  "test_duration": 0.1,
  "tests": [
    {
      "gamma": 1.4,
      "cfl": 0.2,
      "resolution": [48, 48],
      "result": {
        "final_time": 0.08,
        "steps_completed": 10,
        "max_density": 1.2,
        "min_density": 0.9,
        "max_div_b": 1.2e-15
      },
      "error": null,
      "performance": {
        "initialization_time": 0.05,
        "simulation_time": 0.2,
        "steps_per_second": 50.0
      },
      "success": true
    },
    // More test results...
  ],
  "success_rate": 0.75,
  "error_summary": {
    "total_errors": 3,
    "most_common_errors": [
      ["NaN detected in density array", 2],
      ["Memory allocation error", 1]
    ]
  },
  "recommended_config": {
    "gamma": 1.67,
    "cfl": 0.6,
    "resolution": [48, 48]
  }
}
```

## Usage with JavaScript

Here's a JavaScript example showing how to fetch results from the diagnostic endpoint:

```javascript
// Simple function to test the MHD diagnostic endpoint
async function testMHDDiagnostic() {
  const response = await fetch('/api/mhd-diagnostic/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      simulation_type: 'magnetic_rotor',
      resolution: [32, 32]
    }),
  });
  
  const data = await response.json();
  
  if (data.initialization && data.initialization.success) {
    console.log('MHD system initialized successfully');
    console.log('System info:', data.initialization.system_info);
    
    if (data.single_step && data.single_step.success) {
      console.log('Single time step completed successfully');
    } else {
      console.error('Single time step failed:', data.single_step?.error);
    }
    
    if (data.memory_diagnostics) {
      console.log('Total memory usage (MB):', 
        data.memory_diagnostics.memory_usage.total.size_mb);
    }
  } else {
    console.error('MHD initialization failed:', data.initialization?.error);
  }
  
  return data;
}

// Run the test
testMHDDiagnostic().then(results => {
  console.log('Complete diagnostic results:', results);
});
```

## Tips for Troubleshooting MHD Simulations

1. **NaN or Inf values**: Check the `post_step_check` values to see if NaN or Inf values appear after a single step. This often indicates numerical instability.

2. **Memory issues**: Look at the `memory_diagnostics` section to identify potential memory leaks or excessive memory usage.

3. **CFL stability**: Use the stress test to identify the optimal CFL number for your simulation. Higher CFL allows for faster simulations but may lead to instability.

4. **Divergence-free constraint**: The `max_div_b` value should be close to zero. Higher values indicate numerical errors in maintaining the divergence-free constraint of the magnetic field.

5. **Resolution**: If simulations fail at higher resolutions, try running at a lower resolution first to verify that the basic physics is working correctly. 