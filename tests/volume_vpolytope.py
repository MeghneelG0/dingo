import unittest
import numpy as np
import gc
from volestipy import VPolytope

class TestVPolytope(unittest.TestCase):
    
    def setUp(self):
        """Runs before each test to ensure garbage collection."""
        gc.collect()

    def test_2d_square_volume(self):
        """Test volume computation for a 2D square (should be 4.0)."""
        vertices = np.array([
            [-1, -1], [-1, 1],
            [ 1, -1], [ 1, 1]
        ], dtype=np.float64)

        polytope = VPolytope(vertices)
        computed_volume = polytope.compute_volume()

        expected_volume = 4.0
        self.assertAlmostEqual(computed_volume, expected_volume, delta=0.1)

    def test_3d_cube_volume(self):
        """Test volume computation for a 3D cube (should be 8.0)."""
        vertices = np.array([
            [-1, -1, -1], [-1, -1,  1],
            [-1,  1, -1], [-1,  1,  1],
            [ 1, -1, -1], [ 1, -1,  1],
            [ 1,  1, -1], [ 1,  1,  1]
        ], dtype=np.float64)

        polytope = VPolytope(vertices)
        computed_volume = polytope.compute_volume()

        expected_volume = 8.0
        self.assertAlmostEqual(computed_volume, expected_volume, delta=0.2)

    def test_vpolytope_high_dim(self):
        """Test higher dimensions to check for segfaults."""
        dimensions = [4, 5]  # Avoid excessive dimensions to prevent stack overflow

        for d in dimensions:
            with self.subTest(dimension=d):
                # Create a hypercube with vertices at (-1,1)^d
                vertices = np.vstack(np.meshgrid(*([[-1, 1]] * d))).reshape(d, -1).T
                
                polytope = VPolytope(vertices)
                computed_volume = polytope.compute_volume()

                expected_volume = 2**d
                self.assertAlmostEqual(computed_volume, expected_volume, delta=expected_volume * 0.2)

    def test_vpolytope_large_dimension_performance(self):
        """Check volume computation for high dimensions (stress test)."""
        import time
        dimensions = [6]  # Test only up to 6D to avoid crashes

        for d in dimensions:
            with self.subTest(dimension=d):
                vertices = np.vstack(np.meshgrid(*([[-1, 1]] * d))).reshape(d, -1).T
                polytope = VPolytope(vertices)

                start = time.time()
                computed_volume = polytope.compute_volume()
                end = time.time()

                expected_volume = 2**d
                self.assertAlmostEqual(computed_volume, expected_volume, delta=expected_volume * 0.3)
                
                print(f"{d}D computation time: {end - start:.4f} sec")

if __name__ == "__main__":
    unittest.main()
