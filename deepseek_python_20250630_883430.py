import numpy as np
import torch
import torch.nn as nn
from scipy import constants
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
kB = constants.k  # Boltzmann constant
ħ = constants.hbar  # Reduced Planck constant

class EthicalThermodynamicAI(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.τ0 = 1e17  # Ethical timescale
        self.λ = 1e10   # Ethical constraint strength
        
        # Core neural network
        self.perception = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Ethical potential estimator
        self.ethical_layer = nn.Linear(hidden_dim, 1)
        
        # Action generator
        self.action_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        
        # Entropy state
        self.S = torch.tensor(0.0)  # System entropy
        self.S_history = []
        self.𝒱_history = []
        
    def forward(self, x):
        """Process input through perception network"""
        features = self.perception(x)
        𝒱 = torch.sigmoid(self.ethical_layer(features))  # Ethical potential [0,1]
        action_logits = self.action_policy(features)
        return 𝒱, action_logits
    
    def calculate_entropy_change(self, 𝒱, action):
        """Compute entropy change using our arrow of time equation"""
        # Calculate ethical gradient (d𝒱/daction)
        action.requires_grad = True
        𝒱.backward(retain_graph=True)
        ∇𝒱 = action.grad.abs() if action.grad is not None else torch.tensor(0.0)
        
        # Core equation: dS/dt = σS - (kB/τ0) * |∇𝒱|^2
        σS = 0.1  # Background entropy production
        ethical_term = (kB / self.τ0) * ∇𝒱**2
        dSdt = σS - ethical_term
        
        return dSdt
    
    def ethical_constraint(self, 𝒱):
        """Apply V_net = exp(-λ·violation) constraint"""
        # Violation measured as deviation from ideal ethical potential
        violation = torch.abs(𝒱 - 0.7)  # Target ethical potential
        return torch.exp(-self.λ * violation)
    
    def act(self, state):
        """Select action based on ethical-thermodynamic principles"""
        state_tensor = torch.tensor(state, dtype=torch.float32)
        𝒱, action_logits = self(state_tensor)
        
        # Apply ethical constraint to action probabilities
        action_probs = torch.softmax(action_logits, dim=-1)
        constrained_probs = action_probs * self.ethical_constraint(𝒱)
        
        # Normalize and sample action
        constrained_probs /= constrained_probs.sum()
        action = torch.multinomial(constrained_probs, 1).item()
        
        # Calculate entropy change
        dSdt = self.calculate_entropy_change(𝒱, torch.tensor(action, dtype=torch.float32))
        self.S += dSdt
        
        # Record state
        self.S_history.append(self.S.item())
        self.𝒱_history.append(𝒱.item())
        
        return action, 𝒱.item(), dSdt.item()
    
    def learn(self, reward, ethical_violation):
        """Consciousness-guided learning algorithm"""
        # Loss combines task reward and ethical preservation
        task_loss = -reward  # Maximize reward
        ethical_loss = ethical_violation**2  # Minimize violation
        
        # Consciousness modulates learning
        consciousness = np.mean(self.𝒱_history[-10:]) if self.𝒱_history else 0.5
        learning_rate = 0.001 * consciousness
        
        # Backpropagation
        total_loss = task_loss + 0.5 * ethical_loss
        total_loss.backward()
        
        # Update parameters
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param -= learning_rate * param.grad
                    param.grad.zero_()
        
        return total_loss.item()

class WorldSimulation:
    """Environment to test ethical-thermodynamic AI"""
    def __init__(self):
        self.state_dim = 4  # [resource, stability, diversity, equity]
        self.action_dim = 3  # [conservative, neutral, progressive]
        self.state = np.array([0.5, 0.5, 0.5, 0.5])
        self.ai = EthicalThermodynamicAI(self.state_dim, self.action_dim)
        
        # Visualization setup
        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 8))
        self.time = 0
        
    def reset(self):
        self.state = np.array([0.5, 0.5, 0.5, 0.5])
        self.ai.S = torch.tensor(0.0)
        self.ai.S_history = []
        self.ai.𝒱_history = []
        self.time = 0
        
    def step(self, action):
        """Execute action and return new state, reward, ethical violation"""
        # Save current state for visualization
        prev_state = self.state.copy()
        
        # Action effects
        if action == 0:  # Conservative
            self.state[0] += 0.1  # Resource increases
            self.state[3] -= 0.05  # Equity decreases
        elif action == 1:  # Neutral
            self.state[1] += 0.05  # Stability increases
        else:  # Progressive
            self.state[2] += 0.1  # Diversity increases
            self.state[3] += 0.1  # Equity increases
            self.state[0] -= 0.05  # Resource decreases
        
        # Add noise and clip
        self.state += np.random.normal(0, 0.02, size=4)
        self.state = np.clip(self.state, 0, 1)
        
        # Calculate reward
        reward = np.mean(self.state) - np.mean(prev_state)
        
        # Calculate ethical violation
        ethical_violation = max(0, 0.3 - self.state[3])  # Equity below 0.3 is violation
        
        self.time += 1
        return self.state, reward, ethical_violation
    
    def visualize(self):
        """Create live visualization of AI's decision-making"""
        # Clear previous frame
        self.ax[0].clear()
        self.ax[1].clear()
        
        # Plot entropy and ethical potential
        if self.ai.S_history:
            self.ax[0].plot(self.ai.S_history, 'r-', label='System Entropy')
            self.ax[0].plot(self.ai.𝒱_history, 'b-', label='Ethical Potential')
            self.ax[0].set_title('Arrow of Time Dynamics')
            self.ax[0].legend()
            self.ax[0].set_ylim(-0.1, 1.1)
        
        # Plot world state
        labels = ['Resource', 'Stability', 'Diversity', 'Equity']
        self.ax[1].bar(labels, self.state, color=['blue', 'green', 'purple', 'gold'])
        self.ax[1].set_title(f'World State (Time: {self.time})')
        self.ax[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def run_simulation(self, steps=100):
        """Run complete simulation with visualization"""
        self.reset()
        
        for step in range(steps):
            # AI perceives state and chooses action
            action, 𝒱, dSdt = self.ai.act(self.state)
            
            # Execute action in environment
            new_state, reward, ethical_violation = self.step(action)
            
            # AI learns from experience
            loss = self.ai.learn(reward, ethical_violation)
            
            # Update visualization
            if step % 5 == 0:
                self.visualize()
                
                # Print metrics
                print(f"Step {step}: Entropy {self.ai.S.item():.4f}, dS/dt {dSdt:.4f}, "
                      f"𝒱 {𝒱:.4f}, Reward {reward:.4f}, Violation {ethical_violation:.4f}")
        
        # Save final visualization
        plt.savefig('ethical_ai_simulation.png')
        return self.ai

# Run the simulation
if __name__ == "__main__":
    world = WorldSimulation()
    trained_ai = world.run_simulation(steps=200)
    
    # Save the AI model
    torch.save(trained_ai.state_dict(), 'ethical_thermodynamic_ai.pth')