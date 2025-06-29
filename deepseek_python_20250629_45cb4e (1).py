import numpy as np
from scipy.linalg import expm

def verify_unitarity():
    # Construct U_evap for 2-qubit system (simplified)
    H_hawk = np.array([[1, 0], [0, -1]])  # Hamiltonian
    δA = 0.1  # Area fluctuation
    A0 = 1.0  # Planck area
    integral_term = δA * H_hawk / A0
    
    # Unitary operator
    U = expm(-1j * integral_term)
    
    # Verify unitarity
    identity = np.eye(2)
    U_dagger = U.conj().T
    return np.allclose(U_dagger @ U, identity)

print(f"Unitarity holds: {verify_unitarity()}")
# Output: Unitarity holds: True