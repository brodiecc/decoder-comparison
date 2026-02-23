import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parity_checks import create_H
from logicals import create_logical


def test_commutativity(L_values=[3, 5]):
    for L in L_values:
        print(f"\n--- Testing L={L} ---")

        # 1. Generate Parity Checks
        Hx = create_H(L, "x")
        Hz = create_H(L, "z")

        # 2. Generate Logicals
        X_L = create_logical(L, "x")
        Z_L = create_logical(L, "z")

        print(f"Shapes -> Hx: {Hx.shape}, Hz: {Hz.shape}, Logicals: {X_L.shape}")

        # 3. Check X Logical vs Z Checks (Should Commute)
        # Logical X (Vertical) must commute with Z-plaquettes.
        z_commute_syndrome = (Hz @ X_L.T).data % 2
        if np.any(z_commute_syndrome):
            print("FAIL: Logical X anti-commutes with Z-checks (Code failure).")
        else:
            print("PASS: Logical X commutes with all Z-checks.")

        # 4. Check Z Logical vs X Checks (Should Commute)
        # Logical Z (Horizontal) must commute with X-vertices.
        x_commute_syndrome = (Hx @ Z_L.T).data % 2
        if np.any(x_commute_syndrome):
            print("FAIL: Logical Z anti-commutes with X-checks (Code failure).")
        else:
            print("PASS: Logical Z commutes with all X-checks.")

        # 5. Check Logicals vs Each Other (Should Anti-Commute)
        # They must overlap on an odd number of qubits (1).
        logical_overlap = (X_L @ Z_L.T).toarray()[0, 0] % 2
        if logical_overlap == 1:
            print("PASS: Logical X and Z anti-commute (Valid logical qubit).")
        else:
            print(f"FAIL: Logical X and Z commute! (Overlap = {logical_overlap})")


if __name__ == "__main__":
    test_commutativity()
