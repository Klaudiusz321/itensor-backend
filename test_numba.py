import numpy as np
from numba import njit
from numba.typed import List

# Import the function we want to test
from myproject.utils.mhd.constrained_transport import face_to_cell_centered_b

def test_face_to_cell_centered_b():
    # Create sample face-centered B for 2D case
    bx_face = np.ones((5, 4))  # Shape: (nx+1, ny)
    by_face = np.ones((4, 5))  # Shape: (nx, ny+1)
    
    # Create a Numba typed List
    face_b = List()
    face_b.append(bx_face)
    face_b.append(by_face)
    
    try:
        # Try to call the function
        cell_b = face_to_cell_centered_b(face_b)
        print("✓ face_to_cell_centered_b compiled and executed successfully")
        print(f"Output shapes: {cell_b[0].shape}, {cell_b[1].shape}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Numba JIT compilation...")
    test_face_to_cell_centered_b() 