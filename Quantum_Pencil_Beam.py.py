import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running Hi-Res 3D Animation on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

class PyTorchQuantumAntenna3D:
    def __init__(self, N=8, wavelength=1.0, spacing=0.5):
        self.N = N
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
        self.d = spacing * wavelength
        self.device = device
        self.dim = 2**self.N
        self.state = torch.zeros(self.dim, dtype=torch.complex64, device=self.device)

    def prepare_w_state(self):
        """Prepare entangled W-state"""
        self.state = torch.zeros(self.dim, dtype=torch.complex64, device=self.device)
        indices = 2**torch.arange(self.N, device=self.device)
        self.state[indices] = 1.0 / np.sqrt(self.N)

    def steer_beam(self, target_angle):
        """Apply quantum phases to steer the beam"""
        n_indices = torch.arange(self.N, device=self.device)
        phases = n_indices * self.k * self.d * torch.sin(torch.tensor(target_angle))
        
        indices = torch.arange(self.dim, device=self.device)
        total_phase = torch.zeros(self.dim, device=self.device)
        
        for i in range(self.N):
            bit_mask = (indices >> i) & 1
            total_phase += bit_mask * phases[i]
            
        self.state = self.state * torch.exp(1j * total_phase)

    def calculate_3d_pattern(self, res_theta=200, res_phi=200):
        """
        Calculate High-Res 3D intensity surface.
        """
        # High Resolution Grid
        theta = torch.linspace(-np.pi/2, np.pi/2, res_theta, device=self.device)
        phi = torch.linspace(0, 2*np.pi, res_phi, device=self.device)
        
        THETA, PHI = torch.meshgrid(theta, phi, indexing='ij')
        
        # Extract active amplitudes (W-state optimization)
        excitation_indices = 2**torch.arange(self.N, device=self.device)
        amplitudes = self.state[excitation_indices]

        # Vectorized Array Factor Calculation
        sin_thetas = torch.sin(THETA)
        n_vec = torch.arange(self.N, device=self.device).view(1, 1, -1)
        sin_t_expanded = sin_thetas.unsqueeze(-1)
        
        exponents = 1j * n_vec * self.k * self.d * sin_t_expanded
        phase_factors = torch.exp(exponents)
        
        array_factor = torch.matmul(phase_factors, amplitudes)
        intensity = torch.abs(array_factor)**2
        
        if torch.max(intensity) > 0:
            intensity /= torch.max(intensity)
            
        # Convert to Cartesian
        R = intensity
        X = R * torch.sin(THETA) * torch.cos(PHI)
        Y = R * torch.sin(THETA) * torch.sin(PHI)
        Z = R * torch.cos(THETA)
        
        return X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy()

# --- MAIN ANIMATION ROUTINE ---
if __name__ == "__main__":
    # Settings
    N_elements = 8
    frames = 100  # Increased for smoother motion
    res_grid = 200 # Increased mesh resolution
    
    q_gpu = PyTorchQuantumAntenna3D(N=N_elements)
    
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate Array Positions for Visualization
    # We center the array at the origin for the plot
    # Scale spacing visually to fit inside the intensity plot
    visual_spacing = 0.15 
    array_x = np.linspace(-(N_elements-1)*visual_spacing/2, (N_elements-1)*visual_spacing/2, N_elements)
    array_y = np.zeros_like(array_x)
    array_z = np.zeros_like(array_x)

    def update(frame):
        ax.clear()
        
        # Smooth Sine Sweep (-45 to +45 degrees)
        sweep_angle = 45 * np.sin(2 * np.pi * frame / frames)
        
        # Physics Calculation
        q_gpu.prepare_w_state()
        q_gpu.steer_beam(np.deg2rad(sweep_angle))
        X, Y, Z = q_gpu.calculate_3d_pattern(res_theta=res_grid, res_phi=res_grid)
        
        # Plot Radiation Surface (High Res)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, antialiased=True, alpha=0.85, rcount=res_grid, ccount=res_grid)
        
        # Plot The 8-Element Array (Schematic)
        # We plot them as red spheres along the X-axis
        ax.scatter(array_x, array_y, array_z, color='red', s=80, depthshade=False, edgecolors='black', label='8-Qubit Array')
        ax.plot(array_x, array_y, array_z, color='black', linewidth=1) # Connecting line

        # Styling
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(0, 1.0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Intensity')
        
        ax.set_title(f"3D Quantum Phased Array (N={N_elements})", fontsize=18, fontweight='bold')
        ax.text2D(0.05, 0.95, f"Steering Angle: {sweep_angle:.1f}°", transform=ax.transAxes, fontsize=14, color='navy')
        ax.legend(loc='lower right')
        
        # Adjust view for best "3D" feel
        ax.view_init(elev=25, azim=45 + sweep_angle/2) # Subtle camera rotation matching beam

    print("Rendering High-Res 3D Animation... (Please wait)")
    anim = FuncAnimation(fig, update, frames=frames, interval=40, blit=False)
    
    filename = "quantum_antenna_hires_N8.gif"
    anim.save(filename, writer=PillowWriter(fps=25))
    print(f"Success! Saved to {filename}")