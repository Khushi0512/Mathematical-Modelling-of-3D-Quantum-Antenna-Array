import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running 4x4 Planar Array Simulation on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

class PyTorchPlanarArray:
    def __init__(self, Nx=4, Ny=4, wavelength=1.0, spacing=0.5):
        self.Nx = Nx
        self.Ny = Ny
        self.N = Nx * Ny
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
        self.dx = spacing * wavelength
        self.dy = spacing * wavelength
        self.device = device
        self.dim = 2**self.N
        
        # Define Grid Positions (Centered at 0,0)
        x = torch.linspace(-(Nx-1)*self.dx/2, (Nx-1)*self.dx/2, Nx, device=device)
        y = torch.linspace(-(Ny-1)*self.dy/2, (Ny-1)*self.dy/2, Ny, device=device)
        self.grid_x, self.grid_y = torch.meshgrid(x, y, indexing='ij')
        
        # Flatten positions for vectorized calc
        self.pos_x = self.grid_x.flatten()
        self.pos_y = self.grid_y.flatten()
        
        self.state = torch.zeros(self.dim, dtype=torch.complex64, device=self.device)

    def prepare_w_state(self):
        """Prepare entangled W-state for N = Nx * Ny emitters"""
        self.state = torch.zeros(self.dim, dtype=torch.complex64, device=self.device)
        indices = 2**torch.arange(self.N, device=self.device)
        self.state[indices] = 1.0 / np.sqrt(self.N)

    def steer_beam(self, theta_target, phi_target):
        """
        Steer to (theta, phi).
        Phase shift for element at (x,y) is k * sin(theta) * (x*cos(phi) + y*sin(phi))
        """
        # Calculate geometric phases for each emitter position
        # u = sin(theta)*cos(phi), v = sin(theta)*sin(phi)
        sin_t = torch.sin(torch.tensor(theta_target))
        cos_p = torch.cos(torch.tensor(phi_target))
        sin_p = torch.sin(torch.tensor(phi_target))
        
        phase_shifts = self.k * sin_t * (self.pos_x * cos_p + self.pos_y * sin_p)
        
        # Apply to Quantum State
        indices = torch.arange(self.dim, device=self.device)
        total_phase = torch.zeros(self.dim, device=self.device)
        
        for i in range(self.N):
            bit_mask = (indices >> i) & 1
            total_phase += bit_mask * phase_shifts[i]
            
        self.state = self.state * torch.exp(1j * total_phase)

    def calculate_3d_pattern(self, res=150):
        # Grid over sphere
        theta = torch.linspace(0, np.pi/2, res, device=self.device) # 0 to 90 elevation
        phi = torch.linspace(0, 2*np.pi, res, device=self.device)   # 0 to 360 azimuth
        THETA, PHI = torch.meshgrid(theta, phi, indexing='ij')
        
        # Precompute direction vectors
        sin_t = torch.sin(THETA)
        cos_p = torch.cos(PHI)
        sin_p = torch.sin(PHI)
        
        # Array Factor Calculation
        # Phase at observer (theta, phi) for emitter at (x,y)
        # psi = k * sin(theta) * (x*cos(phi) + y*sin(phi))
        
        # Expand dimensions for broadcasting
        # pos_x shape: (N) -> (1, 1, N)
        px = self.pos_x.view(1, 1, -1)
        py = self.pos_y.view(1, 1, -1)
        
        # Expand grid directions: (res, res) -> (res, res, 1)
        st_exp = sin_t.unsqueeze(-1)
        cp_exp = cos_p.unsqueeze(-1)
        sp_exp = sin_p.unsqueeze(-1)
        
        # Exponent: j * k * sin(t) * (x*cos(p) + y*sin(p))
        geometric_phase = st_exp * (px * cp_exp + py * sp_exp)
        exponents = 1j * self.k * geometric_phase
        phase_factors = torch.exp(exponents) # (res, res, N)
        
        # Get active amplitudes (W-state)
        exc_indices = 2**torch.arange(self.N, device=self.device)
        amps = self.state[exc_indices]
        
        # Sum
        array_factor = torch.matmul(phase_factors, amps)
        intensity = torch.abs(array_factor)**2
        
        if torch.max(intensity) > 0:
            intensity /= torch.max(intensity)
            
        # Cartesian conversion for plot
        R = intensity
        X = R * torch.sin(THETA) * torch.cos(PHI)
        Y = R * torch.sin(THETA) * torch.sin(PHI)
        Z = R * torch.cos(THETA)
        
        return X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy()

# --- ANIMATION ---
if __name__ == "__main__":
    Nx, Ny = 4, 4
    q_planar = PyTorchPlanarArray(Nx=Nx, Ny=Ny)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Visual Array Grid (Red dots on XY plane)
    grid_x_cpu = q_planar.pos_x.cpu().numpy()
    grid_y_cpu = q_planar.pos_y.cpu().numpy()
    
    def update(frame):
        ax.clear()
        
        # Spiral Scan Pattern
        t = frame / 50.0
        target_theta = np.deg2rad(30 * np.sin(t)) # Bob up and down 0-30 deg
        target_phi = 2 * np.pi * (frame / 100.0)  # Spin around
        
        # Physics
        q_planar.prepare_w_state()
        q_planar.steer_beam(abs(target_theta), target_phi)
        X, Y, Z = q_planar.calculate_3d_pattern(res=120)
        
        # Plot Surface
        surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, alpha=0.9, antialiased=True)
        
        # Plot Array Grid
        ax.scatter(grid_x_cpu, grid_y_cpu, np.zeros_like(grid_x_cpu), color='red', s=50, label='4x4 Planar Array')
        
        # Limits & Labels
        ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(0, 1)
        ax.set_title(f"2D Quantum Planar Array (Pencil Beam)", fontsize=16)
        ax.text2D(0.05, 0.95, f"Theta: {np.degrees(abs(target_theta)):.1f}°, Phi: {np.degrees(target_phi)%360:.1f}°", 
                  transform=ax.transAxes, color='navy', fontsize=12)
        
    anim = FuncAnimation(fig, update, frames=100, interval=50)
    anim.save("quantum_pencil_beam.gif", writer=PillowWriter(fps=20))
    print("Saved 'quantum_pencil_beam.gif'")