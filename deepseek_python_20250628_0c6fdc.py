def value_alignment(state):
    return np.dot(state, HUMAN_VALUE_VECTOR) > 0.8