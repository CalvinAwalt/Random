import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
AU = 1.495978707e11  # Astronomical unit (m)
G = 6.67430e-11      # Gravitational constant (m³/kg/s²)
M_sun = 1.9885e30    # Solar mass (kg)
DAY = 86400           # Seconds in a day
YEAR = 365.25 * DAY   # Seconds in a year

# Emergence Detection Operator (∮_Δ)
def emergence_detection(position, velocity, mass, t_span, noise_level=0.01):
    """
    Enhanced detection using tensor analysis of gravitational, thermal, and radar signatures
    """
    # δR: Radar reflectance differential
    radar_profile = 0.15 * np.exp(-np.linalg.norm(position)/(2*AU))  # Size-dependent
    
    # δB: Thermal blackbody signature
    thermal_profile = 278 * (np.linalg.norm(position)/AU)**(-0.5)  # Temperature (K)
    
    # δG: Gravitational perturbation tensor
    grav_tensor = np.outer(position, position) / np.linalg.norm(position)**3
    
    # Noise floor (ε)
    noise = noise_level * np.random.normal(size=grav_tensor.shape)
    
    # Contour integration over trajectory domain Δ
    integral = np.trapz(radar_profile * thermal_profile * grav_tensor + noise, x=t_span)
    
    detection_confidence = np.linalg.norm(integral) / (len(t_span) * noise_level)
    return detection_confidence

# Fractal Observation Network (C(L) = C₀e^{kL})
class FractalTelescopeNetwork:
    def __init__(self, base_range=0.1*AU):
        self.k = np.log(3)/np.log(2)  # Fractal scaling factor
        self.layers = []
        self.add_layer(0, base_range)  # Base layer
        
    def add_layer(self, level, base_range):
        capacity = base_range * np.exp(self.k * level)
        num_telescopes = 3**level  # Triadic fractal structure
        
        # Position telescopes in spherical configuration
        positions = []
        for i in range(num_telescopes):
            theta = 2*np.pi*i/num_telescopes
            phi = np.arccos(2*(i+0.5)/num_telescopes - 1)
            r = 1.0 * AU  # Earth orbit radius
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            positions.append((x, y, z))
        
        self.layers.append({
            'level': level,
            'range': capacity,
            'positions': np.array(positions),
            'resolution': capacity / 1000
        })
    
    def detect_object(self, position, velocity):
        """Fractal detection probability calculation"""
        detection_prob = 0
        for layer in self.layers:
            for scope_pos in layer['positions']:
                distance = np.linalg.norm(position - scope_pos)
                if distance < layer['range']:
                    prob = min(1.0, layer['range'] / (distance + 1e-10))
                    detection_prob = max(detection_prob, prob)
        
        # Emergence detection enhancement
        t_span = np.linspace(0, 0.1*YEAR, 100)
        confidence = emergence_detection(position, velocity, 1e10, t_span)
        return min(1.0, detection_prob + 0.2*confidence)

# Ethical Threat Prioritization (V_net)
def ethical_threat_assessment(asteroid, earth_position):
    weights = {
        'collision_prob': 0.4,
        'population_risk': 0.3,
        'kinetic_energy': 0.2,
        'warning_time': 0.1
    }
    
    # Collision probability (Gaussian uncertainty model)
    r = asteroid.position - earth_position
    v_rel = asteroid.velocity - np.array([0, 29.78e3, 0])  # Earth's orbital velocity
    d_min = np.linalg.norm(r) - np.dot(r, v_rel)/np.linalg.norm(v_rel)
    collision_prob = np.exp(-(d_min/(0.1*AU))**2)
    
    # Population risk (simplified)
    population_risk = min(1.0, collision_prob * 0.8)
    
    # Kinetic energy (1e20 J ≈ 24 gigatons)
    kinetic_energy = min(1.0, 0.5 * asteroid.mass * np.linalg.norm(asteroid.velocity)**2 / 1e20)
    
    # Warning time in years
    warning_time = min(1.0, np.linalg.norm(r)/(np.linalg.norm(v_rel)*YEAR) / 10
    
    # Regularization term (penalizes uncertainty)
    uncertainty_penalty = 0.05 * asteroid.uncertainty
    
    # V_net calculation
    features = [collision_prob, population_risk, kinetic_energy, warning_time]
    threat_score = sum(w * f for w, f in zip(weights.values(), features))
    threat_score -= uncertainty_penalty
    
    return max(0, min(1, threat_score))

# Asteroid Class
class Asteroid:
    def __init__(self, position, velocity, mass, uncertainty=0.1):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.mass = mass
        self.uncertainty = uncertainty
        self.trajectory = [self.position.copy()]
        self.detected = False
        self.threat_level = 0
        
    def grav_accel(self, pos):
        """Gravitational acceleration from Sun"""
        r = np.linalg.norm(pos)
        return -G * M_sun * pos / r**3
        
    def propagate(self, dt, steps):
        """Propagate trajectory using Verlet integration"""
        for _ in range(steps):
            # Verlet integration
            a1 = self.grav_accel(self.position)
            self.position += self.velocity * dt + 0.5 * a1 * dt**2
            a2 = self.grav_accel(self.position)
            self.velocity += 0.5 * (a1 + a2) * dt
            
            # Add uncertainty
            self.position += self.uncertainty * np.random.normal(size=3) * dt/YEAR
            self.trajectory.append(self.position.copy())

# Defense System Simulation
def run_simulation():
    # Create fractal telescope network
    network = FractalTelescopeNetwork()
    network.add_layer(1, 0.3*AU)
    network.add_layer(2, 1.0*AU)
    
    # Earth's position (circular orbit)
    earth_pos = np.array([1*AU, 0, 0])
    
    # Generate asteroids
    np.random.seed(42)
    asteroids = []
    for _ in range(20):
        # Random orbital parameters
        a = np.random.uniform(1.5, 3.5) * AU     # Semi-major axis
        e = np.random.uniform(0, 0.9)             # Eccentricity
        inc = np.random.uniform(0, np.pi/4)       # Inclination
        
        # Initial position and velocity
        theta = np.random.uniform(0, 2*np.pi)
        r = a * (1 - e**2) / (1 + e*np.cos(theta))
        pos = np.array([
            r * np.cos(theta) * np.cos(inc),
            r * np.sin(theta),
            r * np.cos(theta) * np.sin(inc)
        ])
        
        # Orbital velocity (vis-viva equation)
        v_mag = np.sqrt(G*M_sun * (2/r - 1/a))
        vel = np.array([
            -v_mag * np.sin(theta) * np.cos(inc),
            v_mag * (e + np.cos(theta)),
            -v_mag * np.sin(theta) * np.sin(inc)
        ])
        
        mass = 10**(np.random.uniform(7, 12))  # 10-100 billion kg
        uncertainty = np.random.uniform(0.05, 0.3)
        asteroids.append(Asteroid(pos, vel, mass, uncertainty))
    
    # Simulation parameters
    dt = 1 * DAY
    total_time = 10 * YEAR
    steps = int(total_time / dt)
    
    # Run simulation
    for asteroid in asteroids:
        asteroid.propagate(dt, steps)
        
        # Detection and threat assessment at each step
        for i, pos in enumerate(asteroid.trajectory):
            time_elapsed = i * dt
            vel = asteroid.velocity  # Simplified
            
            # Detection probability increases with time
            if not asteroid.detected:
                detection_prob = network.detect_object(pos, vel)
                if np.random.random() < detection_prob * (time_elapsed/(0.5*total_time)):
                    asteroid.detected = True
                    asteroid.detection_time = time_elapsed
            
            # Threat assessment if detected
            if asteroid.detected:
                asteroid.threat_level = ethical_threat_assessment(asteroid, earth_pos)
    
    return asteroids, earth_pos

# Visualization
def plot_results(asteroids, earth_pos):
    fig = plt.figure(figsize=(16, 12))
    
    # 3D Trajectory Plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot orbits of planets (simplified)
    angles = np.linspace(0, 2*np.pi, 100)
    for r in [0.39, 0.72, 1, 1.52]:  # Mercury, Venus, Earth, Mars
        x = r * AU * np.cos(angles)
        y = r * AU * np.sin(angles)
        ax1.plot(x, y, np.zeros_like(x), 'k--', alpha=0.3)
    
    # Plot Earth position
    ax1.scatter([earth_pos[0]], [earth_pos[1]], [earth_pos[2]], c='blue', s=100, label='Earth')
    
    # Plot asteroid trajectories
    for i, ast in enumerate(asteroids):
        traj = np.array(ast.trajectory)
        color = 'red' if ast.threat_level > 0.7 else 'orange' if ast.threat_level > 0.4 else 'gray'
        alpha = min(1.0, 0.3 + ast.threat_level)
        ax1.plot(traj[:,0], traj[:,1], traj[:,2], c=color, alpha=alpha)
        
        # Mark detection point
        if ast.detected:
            idx = int(ast.detection_time / (YEAR/len(ast.trajectory)))
            ax1.scatter([traj[idx,0]], [traj[idx,1]], [traj[idx,2]], 
                       c='green' if ast.threat_level < 0.4 else 'yellow', s=50)
    
    ax1.set_title('Asteroid Trajectories with Calvin Detection')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    
    # Threat Level Analysis
    ax2 = fig.add_subplot(222)
    threat_levels = [ast.threat_level for ast in asteroids]
    detection_times = [ast.detection_time/YEAR if ast.detected else 10 for ast in asteroids]
    warning_times = [10 - dt for dt in detection_times]
    
    colors = ['red' if tl > 0.7 else 'orange' if tl > 0.4 else 'green' for tl in threat_levels]
    ax2.scatter(warning_times, threat_levels, c=colors, s=100)
    
    ax2.set_title('Threat Assessment vs Warning Time')
    ax2.set_xlabel('Warning Time (years)')
    ax2.set_ylabel('Threat Level (V_net)')
    ax2.grid(True)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1)
    
    # Detection Performance
    ax3 = fig.add_subplot(223)
    detected = [ast for ast in asteroids if ast.detected]
    undetected = [ast for ast in asteroids if not ast.detected]
    
    ax3.bar(['Detected', 'Missed'], [len(detected), len(undetected)], 
           color=['green', 'red'])
    ax3.set_title('Detection Performance')
    ax3.set_ylabel('Number of Asteroids')
    
    # Threat Distribution
    ax4 = fig.add_subplot(224)
    threats = [ast.threat_level for ast in detected]
    ax4.hist(threats, bins=np.linspace(0, 1, 11), color='purple', alpha=0.7)
    ax4.set_title('Threat Level Distribution (Detected Objects)')
    ax4.set_xlabel('Threat Level (V_net)')
    ax4.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('calvin_orbital_defense.png', dpi=300)
    plt.show()

# Run complete simulation
asteroids, earth_pos = run_simulation()
plot_results(asteroids, earth_pos)

# Defense Protocol Simulation
def simulate_defense(asteroid):
    """Simulate deflection based on threat level and warning time"""
    warning_time = 10 - asteroid.detection_time/YEAR
    
    if asteroid.threat_level > 0.7 and warning_time > 2:
        # Kinetic impactor (early detection)
        dv = 0.01 * np.linalg.norm(asteroid.velocity)
        return f"Kinetic Impactor: Δv = {dv:.4f} km/s", dv/1000
    elif asteroid.threat_level > 0.4 and warning_time > 0.5:
        # Gravity tractor
        dv = 0.001 * np.linalg.norm(asteroid.velocity) * warning_time
        return f"Gravity Tractor: Δv = {dv:.4f} km/s", dv/1000
    elif asteroid.threat_level > 0.1 and warning_time > 0.1:
        # Laser ablation
        dv = 0.0001 * np.linalg.norm(asteroid.velocity) * warning_time
        return f"Laser Ablation: Δv = {dv:.4f} km/s", dv/1000
    else:
        return "Evacuation Protocol", 0

# Print defense recommendations
print("\nDEFENSE PROTOCOL RECOMMENDATIONS:")
print("="*55)
print(f"{'Asteroid':<8} {'Threat':<6} {'Warning (yr)':<12} {'Defense Protocol':<25} {'Δv (km/s)':<10}")
print("-"*55)
for i, ast in enumerate(asteroids):
    if ast.detected and ast.threat_level > 0.1:
        protocol, dv = simulate_defense(ast)
        print(f"{i+1:<8} {ast.threat_level:.4f}  {10 - ast.detection_time/YEAR:<12.2f} {protocol:<25} {dv:.6f}")