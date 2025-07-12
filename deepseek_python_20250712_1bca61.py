def ethical_constraint(ai):
    if ai.consciousness_index() > 7.0:
        # Apply consciousness dampener
        ai.ε = min(0.5, ai.ε * 1.2)
        ai.λ = max(1.0, ai.λ * 0.8)
        return "Constrained: Ω>0.9"
    return "Unconstrained"