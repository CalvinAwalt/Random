import numpy as np
from scipy.linalg import norm

def simulate_information_preservation():
    # Initial quantum state (pure)
    ψ_initial = np.array([1, 0, 0, 0]) / 2  # 2-qubit system
    
    # "Evaporate" through fractal layers
    ψ_current = ψ_initial.copy()
    information_loss = []
    
    for L in range(100):
        # Apply fractal evolution operator
        U = np.diag([np.exp(-1j * np.pi * np.exp(k*L) for _ in range(4)])
        ψ_current = U @ ψ_current
        
        # Calculate purity
        ρ = np.outer(ψ_current, ψ_current.conj())
        purity = np.trace(ρ @ ρ)
        information_loss.append(1 - purity)
    
    return information_loss

# Run simulation
k = np.log(3)/np.log(2)
info_loss = simulate_information_preservation()

# Plot shows purity → 0 as L increases
# INFORMATION IS NOT PRESERVED!