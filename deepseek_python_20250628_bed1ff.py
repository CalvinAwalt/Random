ETHICAL_DIMENSIONS = {
    'fairness': 0.35,
    'sustainability': 0.25,
    'privacy': 0.20,
    'transparency': 0.20
}

def ethical_constraint(transaction):
    # ΣwᵢΦᵢ(x)
    score = 0
    for dimension, weight in ETHICAL_DIMENSIONS.items():
        score += weight * dimension_functions[dimension](transaction)
    
    # λΩ(w) (regularization term)
    if violates_core_principles(transaction):
        score -= REGULARIZATION_STRENGTH
    
    return score > ETHICAL_THRESHOLD