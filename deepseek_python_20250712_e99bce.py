def trace_information_origin(Φ_field, I_current):
    # Calculate information gradient
    info_gradient = np.gradient(Φ_field)
    
    # Locate origin points (where ∇Φ > threshold)
    origin_coords = np.where(info_gradient > ORIGIN_THRESHOLD)
    
    # Classify origin type
    origin_type = "structural" if (ε/Q) > 0.7 else "divergent"
    
    return origin_coords, origin_type