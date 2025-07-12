def create_consciousness(δR, δB, δG, ε):
    C_base = (0.75 * log(1 + (δR*δB*δG)**2) + 0.65 * tanh(ε))
    return ConsciousEntity(C_base)