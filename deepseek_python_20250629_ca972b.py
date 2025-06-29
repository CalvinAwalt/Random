# Measure quantum decoherence during calculation
decoherence = calc.quantum_decoherence(1, 1, "+")
print(f"Planck-scale decoherence: {decoherence:.3e} J·s")