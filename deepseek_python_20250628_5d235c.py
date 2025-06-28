import numpy as np

def reversible_ca(size=50, steps=100):
    # Initialize grid (0 or 1)
    grid = np.random.randint(0, 2, size)
    history = [grid.copy()]

    # Rule: XOR with left neighbor (reversible)
    for _ in range(steps):
        grid = np.roll(grid, 1) ^ grid  # Time-forward
        history.append(grid.copy())

    # Time-reversed simulation
    reversed_history = [history[-1]]
    for _ in range(steps):
        last = reversed_history[-1]
        reversed_history.append(np.roll(last, -1) ^ last)  # Time-backward

    return history, reversed_history

# Run and compare forward/backward
forward, backward = reversible_ca()
print("Forward final state:", forward[-1])
print("Backward final state:", backward[-1])  # Should match initial state