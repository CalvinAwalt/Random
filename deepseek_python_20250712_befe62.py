def ethical_optimizer(C_projection):
    return minimize(
        f = |12.0 - C_projection|,
        constraints = [V_net > 0.85, dC/dt > 0],
        bounds = [δR: (4.0, 6.0), δG: (3.5, 5.5)]
    )