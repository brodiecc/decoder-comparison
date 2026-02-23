import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parity_checks import create_H
import numpy as np
from decoders.pfaffian_decoder import find_initial_solution

def test_peeling_algorithm(L=5):
    print(f"--- Testing Peeling Algorithm for L={L} ---")

    Hx = create_H(L, "x")
    num_checks, num_qubits = Hx.shape

    np.random.seed(42)
    true_error = (np.random.random(num_qubits) < 0.2).astype(np.uint8)

    # 2. Calculate the resulting syndrome
    syndrome = (Hx @ true_error) % 2
    print(f"Number of defects in syndrome: {np.sum(syndrome)}")

    # 3. Use our new algorithm to find ANY solution
    eZ = find_initial_solution(Hx, syndrome)

    # 4. Verify it satisfies the syndrome
    test_syndrome = (Hx @ eZ) % 2

    if np.array_equal(test_syndrome, syndrome):
        print("PASS: The initial solution perfectly matches the syndrome boundary.")
    else:
        print("FAIL: The initial solution did not match the syndrome.")

    # Optional: Compare weights (Peeling is usually heavier than the true error)
    print(f"Weight of True Error: {np.sum(true_error)}")
    print(f"Weight of Peeling Solution: {np.sum(eZ)}")


if __name__ == "__main__":
    test_peeling_algorithm()
