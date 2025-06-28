import numpy as np
from scipy.linalg import expm

# Hamiltonian (energy operator)
H = np.array([[0, 1], [1, 0]], dtype=complex)

# Quantum state (|0> + |1>)/√2
psi = np.array([1, 1], dtype=complex) / np.sqrt(2)

# Time evolution operator (forward + backward)
def evolve(psi, t, direction=1):
    U = expm(-1j * H * t * direction)  # Forward: +t, Backward: -t
    return U @ psi

# Simulate forward and backward simultaneously
time_steps = np.linspace(0, 2*np.pi, 100)
forward_states = [evolve(psi, t, +1) for t in time_steps]
backward_states = [evolve(psi, t, -1) for t in time_steps]

print("Forward final state:", forward_states[-1])
print("Backward final state:", backward_states[-1])  # Should return to initial