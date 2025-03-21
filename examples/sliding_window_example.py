import numpy as np
from volestipy import HPolytope, SlidingWindow

def demo_sliding_window_convergence():
    # Define walk lengths to try
    walk_lengths = [5, 10, 15, 20, 25]
    # Expected volume of a unit cube
    expected_volume = 1.0
    A = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0]], dtype=np.float64)
    b = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    hpoly = HPolytope(A, b)
    
    # Reset sliding window
    window = SlidingWindow(5)
    
    print("\nH-polytope (unit cube) volume approximation:")
    print("--------------------------------------------")
    print(f"{'Walk Length':12s} | {'Volume':10s} | {'Rel. Error to True':10s} | {'Window Error':10s} | {'Window Size':10s}")
    print("-" * 70)
    
    for walk_len in walk_lengths:
        volume = hpoly.compute_volume(
            walk_len=walk_len,
            epsilon=0.1,
            vol_method="cooling_balls",
            walk_method="uniform_ball"
        )
        
        window.push(volume)
        
        # Calculate relative error compared to expected value
        rel_error = abs(volume - expected_volume) / expected_volume
        window_error = window.get_relative_error()
        window_size = window.size()
        
        print(f"{walk_len:12d} | {volume:10.5f} | {rel_error:10.5f} | {window_error:10.5f} | {window_size:10d}")
        
        # Check for convergence using sliding window
        if window_size >= 3 and window_error < 0.05:
            print(f"\nConvergence detected at walk length {walk_len}!")
            print(f"Sliding window error: {window_error:.6f}")
            break
    
    print(f"\nSlidingWindow convergence measure: {window.get_relative_error():.6f}")
    print(f"Current window size: {window.size()}")
    print(f"(This is |newest - oldest|/|newest| from the last {window.size()} volume calculations)")
    print("\nSmaller convergence measure indicates better convergence.")
    print("When the value is below your error tolerance (epsilon),")
    print("you can consider the volume computation converged.")

if __name__ == "__main__":
    demo_sliding_window_convergence() 