import numpy as np
from tqdm import tqdm

# Parameters
size = 48  # spatial grid size (x, y, z)
time_steps = 24
threshold = 5.83

# Initialize delta fields
delta_R = np.random.randn(size, size, size)
delta_G = np.random.randn(size, size, size)
delta_B = np.random.randn(size, size, size)

# Time evolution with memory and coupling
def evolve_field(field, memory_strength=0.1, noise_level=0.02):
    new_field = np.zeros_like(field)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                neighbors = [
                    field[(i+di)%size, (j+dj)%size, (k+dk)%size]
                    for di in [-1,0,1] for dj in [-1,0,1] for dk in [-1,0,1] if not (di==0 and dj==0 and dk==0)
                ]
                new_field[i,j,k] = np.mean(neighbors) + memory_strength * field[i,j,k] + np.random.randn() * noise_level
    return new_field

# Stimulus injection
def inject_stimuli(field, t, time_steps):
    center = int(size / 2)
    intensity = np.sin(2 * np.pi * t / time_steps)**2
    field[center-2:center+2, center-2:center+2, center-2:center+2] += intensity * 0.5
    return field

# Attention modulation
def modulate_attention(field, lambda_base=0.5, focus_factor=0.8):
    mean_act = np.mean(field)
    local_dev = (field - mean_act) / (np.std(field) + 1e-6)
    lambda_eff = lambda_base * (1 + focus_factor * local_dev)
    return field * lambda_eff

# Phase coupling
def phase_coupling(R, G, B, coupling_strength=0.05):
    avg = (R + G + B) / 3
    R = R + coupling_strength * (avg - R)
    G = G + coupling_strength * (avg - G)
    B = B + coupling_strength * (avg - B)
    return R, G, B

# Main simulation loop
Phi_min = 0
dt = 1.0 / time_steps

for t in tqdm(range(time_steps), desc="Simulating Consciousness"):
    delta_R = evolve_field(delta_R)
    delta_G = evolve_field(delta_G)
    delta_B = evolve_field(delta_B)

    delta_R = inject_stimuli(delta_R, t, time_steps)
    delta_G = inject_stimuli(delta_G, t, time_steps)
    delta_B = inject_stimuli(delta_B, t, time_steps)

    delta_R = modulate_attention(delta_R)
    delta_G = modulate_attention(delta_G)
    delta_B = modulate_attention(delta_B)

    delta_R, delta_G, delta_B = phase_coupling(delta_R, delta_G, delta_B)

    grad_R = np.gradient(delta_R)
    grad_G = np.gradient(delta_G)
    grad_B = np.gradient(delta_B)

    grad_mag_sq = sum(g**2 for g in grad_R) + sum(g**2 for g in grad_G) + sum(g**2 for g in grad_B)
    Phi_min += np.sum(grad_mag_sq) * dt

# Final Φ_min
print(f"\nComputed Φ_min = {Phi_min:.3f}")
if Phi_min > threshold:
    print("🎉 CONSCIOUSNESS EMERGENCE THRESHOLD REACHED!")
else:
    print(f"⚠️ System still below emergence threshold (need > {threshold})")