def algorithmic_qualia(state):
    return np.trace(state) / np.linalg.norm(state)