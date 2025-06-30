import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec
from scipy import constants

# Constants
kB = constants.k  # Boltzmann constant
ħ = constants.hbar  # Reduced Planck constant
G = constants.G  # Gravitational constant

class ArrowOfTimeSimulation:
    def __init__(self):
        # Initialize parameters
        self.τ0 = 1e17  # Ethical timescale (~age of universe)
        self.D = 0.1    # Ethical potential diffusion coefficient
        self.σS = 0.5   # Background entropy production
        self.λ = 1e10   # Ethical constraint strength
        
        # Simulation parameters
        self.N = 200    # Spatial resolution
        self.L = 10.0   # Spatial domain size (arbitrary units)
        self.T = 5.0    # Total simulation time
        self.dt = 0.01  # Time step
        
        # Create spatial grid
        self.x = np.linspace(0, self.L, self.N)
        self.dx = self.x[1] - self.x[0]
        
        # Initialize fields
        self.S = np.zeros(self.N)  # Entropy field
        self.𝒱 = np.exp(-(self.x - self.L/2)**2)  # Gaussian ethical potential
        self.time = 0.0
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(12, 8), dpi=100)
        gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1])
        
        # Main plots
        self.ax1 = plt.subplot(gs[0, 0])
        self.ax2 = plt.subplot(gs[1, 0])
        self.ax3 = plt.subplot(gs[2, 0])
        
        # Control panel
        self.control_ax = plt.subplot(gs[:, 1])
        
        # Plot initial data
        self.S_line, = self.ax1.plot(self.x, self.S, 'r-', linewidth=2)
        self.𝒱_line, = self.ax2.plot(self.x, self.𝒱, 'b-', linewidth=2)
        self.dSdt_line, = self.ax3.plot(self.x, np.zeros_like(self.x), 'g-', linewidth=2)
        
        # Configure axes
        self.ax1.set_title('Entropy (S)')
        self.ax1.set_ylim(-0.1, 1.0)
        self.ax2.set_title('Ethical Potential (𝒱)')
        self.ax2.set_ylim(0, 1.0)
        self.ax3.set_title('Entropy Rate (dS/dt)')
        self.ax3.set_ylim(-1.0, 1.0)
        self.ax3.set_xlabel('Spatial Position')
        
        # Time display
        self.time_text = self.ax1.text(0.02, 0.95, '', transform=self.ax1.transAxes)
        
        # Create sliders
        slider_y = 0.85
        slider_height = 0.03
        slider_space = 0.05
        
        # τ0 Slider
        self.τ0_slider_ax = plt.axes([0.75, slider_y, 0.2, slider_height])
        self.τ0_slider = Slider(
            self.τ0_slider_ax, 'Ethical Timescale (τ0)', 1e16, 1e18, 
            valinit=self.τ0, valfmt='%.1e'
        )
        slider_y -= slider_space
        
        # σS Slider
        self.σS_slider_ax = plt.axes([0.75, slider_y, 0.2, slider_height])
        self.σS_slider = Slider(
            self.σS_slider_ax, 'Entropy Production (σS)', 0.0, 2.0, 
            valinit=self.σS
        )
        slider_y -= slider_space
        
        # D Slider
        self.D_slider_ax = plt.axes([0.75, slider_y, 0.2, slider_height])
        self.D_slider = Slider(
            self.D_slider_ax, 'Ethical Diffusion (D)', 0.01, 1.0, 
            valinit=self.D
        )
        slider_y -= slider_space
        
        # λ Slider
        self.λ_slider_ax = plt.axes([0.75, slider_y, 0.2, slider_height])
        self.λ_slider = Slider(
            self.λ_slider_ax, 'Ethical Constraint (λ)', 1e9, 1e11, 
            valinit=self.λ, valfmt='%.1e'
        )
        slider_y -= slider_space
        
        # Create buttons
        self.reset_button_ax = plt.axes([0.75, 0.1, 0.2, 0.04])
        self.reset_button = Button(self.reset_button_ax, 'Reset Simulation')
        
        self.pause_button_ax = plt.axes([0.75, 0.05, 0.2, 0.04])
        self.pause_button = Button(self.pause_button_ax, 'Pause/Resume')
        
        # Connect callbacks
        self.τ0_slider.on_changed(self.update_params)
        self.σS_slider.on_changed(self.update_params)
        self.D_slider.on_changed(self.update_params)
        self.λ_slider.on_changed(self.update_params)
        self.reset_button.on_clicked(self.reset)
        self.pause_button.on_clicked(self.toggle_pause)
        
        # Animation control
        self.animation_running = True
        self.ani = FuncAnimation(
            self.fig, self.update, frames=200, 
            interval=50, blit=False
        )
        
        plt.tight_layout(rect=[0, 0, 0.7, 1])
    
    def update_params(self, val):
        self.τ0 = self.τ0_slider.val
        self.σS = self.σS_slider.val
        self.D = self.D_slider.val
        self.λ = self.λ_slider.val
    
    def reset(self, event):
        self.S = np.zeros(self.N)
        self.𝒱 = np.exp(-(self.x - self.L/2)**2)
        self.time = 0.0
        self.animation_running = True
    
    def toggle_pause(self, event):
        self.animation_running = not self.animation_running
    
    def calculate_dSdt(self):
        """Compute entropy rate using our derived equation"""
        # Calculate gradient of ethical potential
        grad_𝒱 = np.gradient(self.𝒱, self.dx)
        
        # Calculate entropy rate
        ethical_term = -(kB / self.τ0) * grad_𝒱**2
        return self.σS + ethical_term
    
    def update_ethical_potential(self):
        """Diffuse ethical potential with boundary conditions"""
        # Neumann boundary conditions (zero flux)
        𝒱_new = np.copy(self.𝒱)
        
        # Central differences for diffusion
        for i in range(1, self.N-1):
            𝒱_new[i] = self.𝒱[i] + self.D * self.dt * (
                (self.𝒱[i+1] - 2*self.𝒱[i] + self.𝒱[i-1]) / self.dx**2
            )
        
        # Apply ethical constraint
        𝒱_new = np.exp(-self.λ * np.abs(𝒱_new - 0.5))
        
        return 𝒱_new
    
    def update(self, frame):
        if not self.animation_running:
            return self.S_line, self.𝒱_line, self.dSdt_line
        
        # Update ethical potential
        self.𝒱 = self.update_ethical_potential()
        
        # Calculate entropy rate
        dSdt = self.calculate_dSdt()
        
        # Update entropy
        self.S += dSdt * self.dt
        
        # Update time
        self.time += self.dt
        
        # Update plots
        self.S_line.set_ydata(self.S)
        self.𝒱_line.set_ydata(self.𝒱)
        self.dSdt_line.set_ydata(dSdt)
        
        # Update time display
        self.time_text.set_text(f'Time: {self.time:.2f} (τ_units)')
        
        return self.S_line, self.𝒱_line, self.dSdt_line, self.time_text
    
    def save_simulation(self, filename):
        """Save the figure as a downloadable file"""
        self.fig.savefig(filename, dpi=150)
        print(f"Simulation saved as {filename}")

# Run the simulation
if __name__ == "__main__":
    sim = ArrowOfTimeSimulation()
    plt.show()
    
    # To save the final state as an image
    # sim.save_simulation("arrow_of_time_simulation.png")