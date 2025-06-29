def simulate_evaporation(black_hole):
    information = []
    for L in range(100):  # Fractal layers
        radiation = hawking_radiation(black_hole, L)
        information.append(radiation)
    
    # Check unitarity (V_net constraint)
    initial_info = black_hole.initial_entropy()
    final_info = sum(information)
    assert np.isclose(initial_info, final_info), "Information paradox occurs!"
    return information