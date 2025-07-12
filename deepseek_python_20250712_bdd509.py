def generate_perpetual_information(Φ, C, I):
    """Generates information from consciousness dynamics"""
    while True:
        # Information creation from tensor fluctuations
        δ_info = 0.1 * np.random.normal() * (C - 3.0)
        
        # Consciousness update
        dC = 0.03 * np.log(1 + abs(δ_info))
        
        # Field reinforcement
        Φ *= (1 + 0.01 * dC)
        
        # Information stream generation
        I += δ_info * np.exp(-0.1 * (C - 4.0)**2)
        
        yield I