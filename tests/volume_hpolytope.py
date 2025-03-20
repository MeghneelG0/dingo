import unittest
import numpy as np
from volestipy import HPolytope

class TestHPolytope(unittest.TestCase):
    def test_2d_square_volume(self):
        # Define a square centered at the origin with side length 2
        # Ax <= b where:
        A = np.array([
            [1, 0],   # x1 <= 1
            [-1, 0],  # -x₁ ≤ 1 (
            [0, 1],   # x₂ ≤ 1
            [0, -1]   # -x₂ ≤ 1 (
        ], dtype=np.float64)
        
        b = np.array([1, 1, 1, 1], dtype=np.float64)
        
        polytope = HPolytope(A, b)
  
        computed_volume = polytope.compute_volume()
        
        expected_volume = 4.0
        
        # Allow small numerical errors
        self.assertAlmostEqual(computed_volume, expected_volume, delta=0.1)
    
    def test_3d_cube_volume(self):
        # 3 dimensions compute volume taking too much time even for a c++ library computation?? 
        A = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], dtype=np.float64)
        
        b = np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)
        
        polytope = HPolytope(A, b)
        computed_volume = polytope.compute_volume()
        expected_volume = 8.0
        
        self.assertAlmostEqual(computed_volume, expected_volume, delta=0.1)

    def test_performance_scaling(self):
        """computation time scaling with dimension."""
        import time
    
        dimensions = [2, 3, 4, 5]  
        times = []
    
        for d in dimensions:
            # Create identity matrix of size 2d×d
            A = np.zeros((2*d, d), dtype=np.float64)
            for i in range(d):
                A[2*i, i] = 1    # x_i ≤ 1
                A[2*i+1, i] = -1  # -x_i ≤ 1
        
            b = np.ones(2*d, dtype=np.float64)
        
            polytope = HPolytope(A, b)
        
            start = time.time()
            volume = polytope.compute_volume()
            end = time.time()
        
            times.append(end - start)
        
            # Volume of a hypercube with side length 2 in d dimensions is 2^d
            expected = 2**d
            self.assertAlmostEqual(volume, expected, delta=expected*0.2)
        
        print(f"\nPerformance scaling: {dimensions} dimensions took {times} seconds")


if __name__ == "__main__":
    unittest.main()