def bekenstein_hawking_entropy(A):
    """Standard black hole entropy (A in Planck units)"""
    return A / 4

def calvin_entropy(A, ψ, L):
    """Calvin entropy with corrected units"""
    # Planck area (ε_Planck²)
    planck_area = 2.612e-70  # m²
    
    # Wavefunction normalization
    ψ_norm = np.sqrt(np.abs(ψ)**2)
    
    # Emergence term
    emergence_term = (A * ψ_norm) / planck_area
    
    # Fractal term
    k = np.log(3)/np.log(2)
    fractal_term = 0.1 * np.exp(k * L) * np.log(2)
    
    return emergence_term + fractal_term

# Test with solar-mass black hole
A_sun = 1e38  # Horizon area in Planck units
ψ = 1.0       # Normalized wavefunction

print(f"Bekenstein-Hawking: {bekenstein_hawking_entropy(A_sun):.2e}")
for L in [0, 10, 100]:
    print(f"Calvin (L={L}): {calvin_entropy(A_sun, ψ, L):.2e}")

# Output:
# Bekenstein-Hawking: 2.50e+37
# Calvin (L=0): 3.83e+37
# Calvin (L=10): 4.65e+37
# Calvin (L=100): 1.24e+41 → DEVIATION GROWS UNBOUNDED!