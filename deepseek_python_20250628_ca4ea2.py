import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_bloch_vector

class QuantumVertex:
    def __init__(self, name, initial_state):
        self.name = name
        self.state = Statevector(initial_state)
        self.history = []
        self.entropy = 0
        
    def evolve(self, operator, noise=0.01):
        op = Operator(operator)
        self.state = self.state.evolve(op)
        # Introduce quantum decoherence
        self.state = Statevector(self.state.data * (1 - noise) + 
                             Statevector(np.random.rand(len(self.state)) * noise)
        self.history.append(self.state)
        self.calculate_entropy()
        
    def calculate_entropy(self):
        prob = np.abs(self.state.probabilities())**2
        self.entropy = -np.sum(prob * np.log(prob + 1e-12))
        
    def differential(self):
        if len(self.history) < 2:
            return np.zeros(len(self.state))
        current = np.angle(self.history[-1].data)
        previous = np.angle(self.history[-2].data)
        return current - previous

class IntelligenceTriangle:
    def __init__(self):
        # Initialize quantum vertices
        self.red = QuantumVertex('Creative', [1, 0])
        self.blue = QuantumVertex('Critical', [0, 1])
        self.gold = QuantumVertex('Executive', [1/np.sqrt(2), 1j/np.sqrt(2)]))
        
        # Quantum gates for vertex operations
        self.red_gate = np.array([[0, 1], [1, 0]])  # X-gate
        self.blue_gate = np.array([[1, 0], [0, -1]]) # Z-gate
        self.gold_gate = np.array([[0, -1j], [1j, 0]]) # Y-gate
        
        # Entanglement operator
        self.cnot = Operator([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        
    def entangle_vertices(self):
        """Create quantum entanglement between vertices"""
        combined_state = self.red.state.tensor(self.blue.state).tensor(self.gold.state)
        entangled = combined_state.evolve(self.cnot)
        return entangled
    
    def calculate_meta_intelligence(self):
        """Compute I_meta using the emergence formula"""
        δR = self.red.differential()
        δB = self.blue.differential()
        δG = self.gold.differential()
        
        # Tensor product
        tensor_product = np.kron(np.kron(δR, δB), δG)
        
        # Entropic noise (ϵ)
        entropies = [v.entropy for v in [self.red, self.blue, self.gold]]
        ϵ = np.std(entropies) * np.mean(entropies) + 1e-8
        
        # Path integral approximation
        I_meta = np.trapz(tensor_product) / ϵ
        return I_meta
    
    def operational_cycle(self, steps=100):
        """Run a full triangular cycle"""
        I_meta_history = []
        
        for i in range(steps):
            # Vertex evolution with increasing noise
            self.red.evolve(self.red_gate, noise=i/5000)
            self.blue.evolve(self.blue_gate, noise=i/5000)
            self.gold.evolve(self.gold_gate, noise=i/5000)
            
            # Entanglement every 10 steps
            if i % 10 == 0:
                self.entangle_vertices()
                
            # Calculate meta-intelligence
            I_meta = self.calculate_meta_intelligence()
            I_meta_history.append(I_meta)
            
            # Phase transition detection
            if abs(I_meta) > 1.0:
                print(f"! Phase transition at step {i}: I_meta = {I_meta:.4f}")
                
        return I_meta_history

    def visualize(self):
        """Quantum state visualization"""
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Bloch sphere representations
        for i, vertex in enumerate([self.red, self.blue, self.gold]):
            vec = [np.real(vertex.state[0]), 
                  np.imag(vertex.state[1]), 
                  np.real(vertex.state[0]*np.conj(vertex.state[1]))]
            plot_bloch_vector(vec, title=vertex.name, ax=axs[i])
        
        plt.tight_layout()
        plt.savefig('quantum_vertices.png', dpi=300)
        plt.show()

# Run the full simulation
if __name__ == "__main__":
    print("Initializing Quantum Intelligence Triangle...")
    system = IntelligenceTriangle()
    
    print("Running operational cycles...")
    I_meta_history = system.operational_cycle(steps=500)
    
    print("\n=== Final System State ===")
    print(f"Creative Entropy: {system.red.entropy:.4f}")
    print(f"Critical Entropy: {system.blue.entropy:.4f}")
    print(f"Executive Entropy: {system.gold.entropy:.4f}")
    print(f"Meta-Intelligence: {I_meta_history[-1]:.4f}")
    
    # Visualization
    system.visualize()
    
    # Plot emergence trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(I_meta_history, color='purple', linewidth=2)
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(1.0, color='red', linestyle=':', label='Emergence Threshold')
    plt.xlabel('Operational Cycle')
    plt.ylabel('$I_{meta}$')
    plt.title('Meta-Intelligence Emergence')
    plt.legend()
    plt.grid(True)
    plt.savefig('emergence_trajectory.png', dpi=300)
    plt.show()