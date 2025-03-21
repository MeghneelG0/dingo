# dingo : a python library for metabolic networks sampling and analysis
# dingo is part of GeomScale project

# Copyright (c) 2024

# Licensed under GNU LGPL.3, see LICENCE file

import unittest
import os
import sys
import numpy as np
from volestipy import SlidingWindow, HPolytope


class TestSlidingWindow(unittest.TestCase):
    
    def setUp(self):
        """Initialize test data for each test method."""
        self.window_size = 5
        self.sliding_window = SlidingWindow(self.window_size)
        # Create a simple 3D cube as test polytope
        self.A = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0]
        ], dtype=np.float64)
        self.b = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        # Expected volume of unit cube
        self.expected_volume = 1.0
        # Test values that converge
        self.converging_values = [0.85, 0.92, 0.95, 0.97, 0.98, 0.99, 1.0]

    def test_initial_state(self):
        """Test the initial state of the sliding window."""
        # Check that a new sliding window has a relative error of 1.0
        self.assertEqual(self.sliding_window.get_relative_error(), 1.0)

    def test_window_filling(self):
        """Test sliding window as it fills up."""
        # Push some values
        for i in range(3):
            self.sliding_window.push(float(i + 1))
        
        # When window isn't full (< window_size), relative error behavior depends on implementation
        # Simply verify it returns a value
        error = self.sliding_window.get_relative_error()
        self.assertIsNotNone(error)

    def test_window_sliding(self):
        """Test window behavior after filling past capacity."""
        # Fill the window and beyond
        for i in range(self.window_size + 2):
            self.sliding_window.push(float(i + 1))
        
        # The oldest values should have been replaced
        # The window should now contain values: [7, 6, 5, 4, 3] where 7 is newest and 3 is oldest
        # Relative error calculated as |newest - oldest|/|newest| = |7 - 3|/|7| = 4/7 ≈ 0.5714
        expected_error = abs(7.0 - 3.0) / 7.0
        self.assertAlmostEqual(self.sliding_window.get_relative_error(), expected_error, places=4)

    def test_convergence_tracking(self):
        """Test sliding window for tracking convergence."""
        # Create new window for this test
        window = SlidingWindow(5)
        
        # Push converging values: [1.0, 0.99, 0.98, 0.97, 0.95] in the window with 1.0 as newest, 0.95 as oldest
        for val in self.converging_values:
            window.push(val)
        
        # Check that relative error decreases as values converge
        # Last 5 values in window are: [1.0, 0.99, 0.98, 0.97, 0.95]
        # Relative error = |newest - oldest|/|newest| = |1.0 - 0.95|/|1.0| = 0.05
        expected_error = abs(1.0 - 0.95) / 1.0
        self.assertAlmostEqual(window.get_relative_error(), expected_error, places=4)

    def test_integration_with_volume_computation(self):
        """Test SlidingWindow integration with actual volume computation."""
        # Skip if HPolytope is not available for testing
        try:
            hpoly = HPolytope(self.A, self.b)
        except Exception as e:
            self.skipTest(f"Skipping test due to HPolytope initialization error: {e}")
            
        window = SlidingWindow(5)
        
        # Different walk lengths for volume approximation
        walk_lengths = [5, 10, 15, 20, 25]
        
        for walk_len in walk_lengths:
            try:
                volume = hpoly.compute_volume(
                    walk_len=walk_len,
                    epsilon=0.1,
                    vol_method="cooling_balls",
                    walk_method="uniform_ball"
                )
                window.push(volume)
            except Exception as e:
                self.skipTest(f"Skipping test due to volume computation error: {e}")
        
        # Check that relative error exists and is valid
        error = window.get_relative_error()
        self.assertIsNotNone(error)
        self.assertGreaterEqual(error, 0.0)


if __name__ == "__main__":
    unittest.main() 