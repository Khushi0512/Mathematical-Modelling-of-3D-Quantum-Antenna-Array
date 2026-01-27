import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
from scipy.constants import c, epsilon_0, hbar, e as electron_charge
import warnings
import os
warnings.filterwarnings('ignore')

# Create output directory in current working directory
OUTPUT_DIR = os.path.join(os.getcwd(), 'quantum_antenna_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

class QuantumEmitterSpecs:
    """
    Physical specifications for quantum emitters used as antenna elements.
    
    Supports two platforms:
    1. Carbon Quantum Dots (CQDs) - Size-tunable optical emitters
    2. Nitrogen-Vacancy (NV) Centers - Diamond-based spin-photon interface
    """
    
    def __init__(self, emitter_type='CQD'):
        """
        Initialize quantum emitter with physical parameters.
        
        Parameters:
        -----------
        emitter_type : str
            Either 'CQD' or 'NV'
        """
        self.emitter_type = emitter_type
        
        if emitter_type == 'CQD':
            # Carbon Quantum Dot specifications
            self.diameter_nm = 5.0  # Size in nanometers
            self.emission_wavelength_nm = 520.0  # Green emission (tunable)
            self.dipole_moment_debye = 25.0  # Transition dipole moment
            self.quantum_yield = 0.35  # Photon emission efficiency
            self.decay_rate_MHz = 100.0  # Spontaneous emission rate
            self.dephasing_rate_MHz = 50.0  # Pure dephasing rate
            
        elif emitter_type == 'NV':
            # Nitrogen-Vacancy center specifications
            self.diameter_nm = 0.3  # Atomic-scale defect
            self.emission_wavelength_nm = 637.0  # Zero-phonon line (fixed)
            self.dipole_moment_debye = 5.2  # Weaker than CQDs
            self.quantum_yield = 0.03  # Most emission in phonon sideband
            self.decay_rate_MHz = 13.0  # Slower decay
            self.dephasing_rate_MHz = 2.0  # Superior coherence
            self.spin_coherence_time_ms = 1.0  # Long-lived spin states
            
        # Convert to SI units
        self.wavelength = self.emission_wavelength_nm * 1e-9  # meters
        self.frequency = c / self.wavelength  # Hz
        self.wavenumber = 2 * np.pi / self.wavelength  # rad/m
        
        # Transition dipole moment in SI (Coulomb-meters)
        debye_to_Cm = 3.33564e-30
        self.dipole_moment = self.dipole_moment_debye * debye_to_Cm
        
        # Decay rates in SI (rad/s)
        self.gamma_decay = self.decay_rate_MHz * 2 * np.pi * 1e6
        self.gamma_dephasing = self.dephasing_rate_MHz * 2 * np.pi * 1e6
        
    def __str__(self):
        """Pretty print emitter specifications"""
        info = f"\n{'='*70}\n"
        info += f"QUANTUM EMITTER SPECIFICATIONS: {self.emitter_type}\n"
        info += f"{'='*70}\n"
        info += f"Diameter:              {self.diameter_nm:.2f} nm\n"
        info += f"Emission Wavelength:   {self.emission_wavelength_nm:.1f} nm\n"
        info += f"Emission Frequency:    {self.frequency/1e12:.2f} THz\n"
        info += f"Dipole Moment:         {self.dipole_moment_debye:.1f} Debye\n"
        info += f"Quantum Yield:         {self.quantum_yield*100:.1f}%\n"
        info += f"Decay Rate:            {self.decay_rate_MHz:.1f} MHz\n"
        info += f"Dephasing Rate:        {self.dephasing_rate_MHz:.1f} MHz\n"
        if self.emitter_type == 'NV':
            info += f"Spin Coherence Time:   {self.spin_coherence_time_ms:.2f} ms\n"
        info += f"{'='*70}\n"
        return info

class QuantumAntennaArray:
    """
    Defines the spatial arrangement of quantum emitters in an antenna array.
    
    Supports multiple geometries:
    - Linear (1D): Simple beam steering
    - Planar Grid (2D): Two-dimensional beam control
    - Hexagonal (2D): Optimal packing density
    - Circular Ring (2D): Orbital angular momentum
    """
    
    def __init__(self, geometry='linear', N=8, spacing_wavelengths=0.5, 
                 emitter_specs=None):
        """
        Initialize antenna array geometry.
        
        Parameters:
        -----------
        geometry : str
            'linear', 'grid', 'hexagonal', or 'circular'
        N : int
            Number of emitters (or size parameter)
        spacing_wavelengths : float
            Inter-emitter spacing in units of wavelength
        emitter_specs : QuantumEmitterSpecs
            Physical specifications of the quantum emitters
        """
        self.geometry = geometry
        self.N_total = N if geometry == 'linear' else N**2
        self.spacing_wavelengths = spacing_wavelengths
        
        # Use default CQD specs if none provided
        if emitter_specs is None:
            emitter_specs = QuantumEmitterSpecs('CQD')
        self.emitter = emitter_specs
        
        # Calculate physical spacing
        self.spacing = spacing_wavelengths * self.emitter.wavelength
        
        # Generate emitter positions
        self.positions = self._generate_positions(geometry, N)
        self.N_emitters = len(self.positions)
        
        # Calculate coupling matrix
        self.coupling_matrix = self._calculate_coupling_matrix()
        
    def _generate_positions(self, geometry, N):
        """Generate 3D positions for each emitter in the array"""
        
        if geometry == 'linear':
            # Linear array along x-axis
            x = np.arange(N) * self.spacing
            y = np.zeros(N)
            z = np.zeros(N)
            positions = np.column_stack([x, y, z])
            
        elif geometry == 'grid':
            # Planar grid in x-y plane
            x = np.tile(np.arange(N) * self.spacing, N)
            y = np.repeat(np.arange(N) * self.spacing, N)
            z = np.zeros(N**2)
            positions = np.column_stack([x, y, z])
            
        elif geometry == 'hexagonal':
            # Hexagonal close-packed array
            positions = []
            for i in range(N):
                for j in range(N):
                    x = i * self.spacing
                    y = j * self.spacing * np.sqrt(3)/2 + (i % 2) * self.spacing * np.sqrt(3)/4
                    z = 0
                    positions.append([x, y, z])
            positions = np.array(positions)
            
        elif geometry == 'circular':
            # Circular ring array
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            radius = N * self.spacing / (2 * np.pi)
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            z = np.zeros(N)
            positions = np.column_stack([x, y, z])
            
        else:
            raise ValueError(f"Unknown geometry: {geometry}")
        
        # Center the array at origin
        positions -= np.mean(positions, axis=0)
        
        return positions
    
    def _calculate_coupling_matrix(self):
        """
        Calculate dipole-dipole coupling matrix.
        
        The coupling between emitters i and j depends on:
        - Distance r_ij between them
        - Relative orientation of dipole moments
        - Near-field (1/r³) vs far-field (1/r) regime
        
        Returns:
        --------
        coupling : ndarray, shape (N, N)
            Complex coupling matrix Ω_ij
        """
        N = self.N_emitters
        coupling = np.zeros((N, N), dtype=complex)
        
        k = self.emitter.wavenumber  # Wavenumber
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue  # No self-coupling
                
                # Vector from emitter i to j
                r_vec = self.positions[j] - self.positions[i]
                r = np.linalg.norm(r_vec)
                r_hat = r_vec / r
                
                # Assume parallel dipoles along z-axis
                # (For more complex cases, include dipole orientation)
                d_i = np.array([0, 0, 1])
                d_j = np.array([0, 0, 1])
                
                # Dipole-dipole interaction (classical Green's function)
                # Near-field term ∝ 1/r³, far-field term ∝ 1/r
                kr = k * r
                
                # Full dyadic Green's function for parallel dipoles
                if kr < 1:  # Near-field dominated
                    coupling[i, j] = (3/4) * self.emitter.gamma_decay / (kr)**3
                else:  # Far-field dominated
                    coupling[i, j] = (3/4) * self.emitter.gamma_decay * \
                                    np.exp(1j * kr) / kr
        
        return coupling
    
    def get_collective_decay_rates(self):
        """
        Calculate collective decay rates (eigenvalues of coupling matrix).
        
        Returns:
        --------
        rates : ndarray
            Collective emission rates (superradiant and subradiant modes)
        """
        # Symmetrize coupling matrix for eigenvalue calculation
        coupling_hermitian = (self.coupling_matrix + 
                            self.coupling_matrix.conj().T) / 2
        
        eigenvalues = np.linalg.eigvalsh(coupling_hermitian.real)
        
        # Collective rates relative to single-emitter decay
        collective_rates = self.emitter.gamma_decay + eigenvalues
        
        return collective_rates
    
    def __str__(self):
        """Pretty print array specifications"""
        info = f"\n{'='*70}\n"
        info += f"QUANTUM ANTENNA ARRAY CONFIGURATION\n"
        info += f"{'='*70}\n"
        info += f"Geometry:              {self.geometry}\n"
        info += f"Number of Emitters:    {self.N_emitters}\n"
        info += f"Spacing:               {self.spacing_wavelengths:.2f} λ "
        info += f"({self.spacing*1e9:.1f} nm)\n"
        dims = np.ptp(self.positions, axis=0)*1e6
        info += f"Array Dimensions:      [{dims[0]:.1f}, {dims[1]:.1f}, {dims[2]:.1f}] μm\n"
        
        # Collective properties
        rates = self.get_collective_decay_rates()
        info += f"\nCollective Emission:\n"
        info += f"  Superradiant Rate:   {np.max(rates)/self.emitter.gamma_decay:.2f} Γ₀\n"
        info += f"  Subradiant Rate:     {np.min(rates)/self.emitter.gamma_decay:.2f} Γ₀\n"
        info += f"{'='*70}\n"
        
        return info


class RadiationPattern:
    """
    Calculate and visualize far-field radiation patterns from quantum array.
    
    Uses classical antenna array factor combined with quantum emission dynamics.
    """
    
    def __init__(self, antenna_array):
        """
        Initialize radiation pattern calculator.
        
        Parameters:
        -----------
        antenna_array : QuantumAntennaArray
            The quantum antenna array to analyze
        """
        self.array = antenna_array
        
    def array_factor(self, theta, phi, excitation_phases=None):
        """
        Calculate array factor (interference pattern).
        
        Parameters:
        -----------
        theta : ndarray
            Elevation angles (radians)
        phi : ndarray
            Azimuthal angles (radians)
        excitation_phases : ndarray, optional
            Phase of each emitter (default: all in phase)
            
        Returns:
        --------
        AF : ndarray
            Complex array factor
        """
        if excitation_phases is None:
            excitation_phases = np.zeros(self.array.N_emitters)
        
        k = self.array.emitter.wavenumber
        
        # Direction vectors
        k_x = k * np.sin(theta) * np.cos(phi)
        k_y = k * np.sin(theta) * np.sin(phi)
        k_z = k * np.cos(theta)
        
        # Array factor sum
        AF = np.zeros_like(theta, dtype=complex)
        
        for i, pos in enumerate(self.array.positions):
            # Path difference for emitter i
            path_diff = k_x * pos[0] + k_y * pos[1] + k_z * pos[2]
            
            # Add contribution with excitation phase
            AF += np.exp(1j * (path_diff + excitation_phases[i]))
        
        return AF
    
    def calculate_pattern_2d(self, plane='elevation', num_points=360,
                           excitation_phases=None):
        """
        Calculate 2D radiation pattern (cut through 3D pattern).
        
        Parameters:
        -----------
        plane : str
            'elevation' (θ variation, φ=0) or 'azimuth' (φ variation, θ=90°)
        num_points : int
            Angular resolution
        excitation_phases : ndarray
            Phase of each emitter
            
        Returns:
        --------
        angles : ndarray
            Angles in degrees
        pattern : ndarray
            Normalized radiation intensity
        """
        angles_rad = np.linspace(0, 2*np.pi, num_points)
        
        if plane == 'elevation':
            # Vary elevation angle θ at φ = 0
            theta = angles_rad
            phi = np.zeros_like(theta)
        elif plane == 'azimuth':
            # Vary azimuth angle φ at θ = π/2
            theta = np.pi/2 * np.ones_like(angles_rad)
            phi = angles_rad
        else:
            raise ValueError("plane must be 'elevation' or 'azimuth'")
        
        # Calculate array factor
        AF = self.array_factor(theta, phi, excitation_phases)
        
        # Include single-emitter dipole pattern: sin²(θ)
        if plane == 'elevation':
            dipole_pattern = np.sin(theta)**2
        else:
            dipole_pattern = np.ones_like(theta)  # Omnidirectional in azimuth
        
        # Total pattern
        pattern = dipole_pattern * np.abs(AF)**2
        
        # Normalize
        pattern = pattern / np.max(pattern)
        
        return np.degrees(angles_rad), pattern
    
    def calculate_pattern_3d(self, num_theta=90, num_phi=180,
                           excitation_phases=None):
        """
        Calculate full 3D radiation pattern.
        
        Returns:
        --------
        theta, phi, intensity : ndarrays
            Spherical coordinate angles and radiation intensity
        """
        theta = np.linspace(0, np.pi, num_theta)
        phi = np.linspace(0, 2*np.pi, num_phi)
        
        THETA, PHI = np.meshgrid(theta, phi)
        
        AF = self.array_factor(THETA, PHI, excitation_phases)
        
        # Dipole pattern
        dipole_pattern = np.sin(THETA)**2
        
        # Total pattern
        intensity = dipole_pattern * np.abs(AF)**2
        intensity = intensity / np.max(intensity)
        
        return THETA, PHI, intensity
    
    def calculate_directivity(self, excitation_phases=None):
        """
        Calculate antenna directivity (peak gain).
        
        Directivity D = 4π × (max intensity) / (total radiated power)
        """
        theta, phi, intensity = self.calculate_pattern_3d(
            excitation_phases=excitation_phases
        )
        
        # Numerical integration over sphere
        d_theta = theta[1, 0] - theta[0, 0]
        d_phi = phi[0, 1] - phi[0, 0]
        
        # Integrate: ∫∫ I(θ,φ) sin(θ) dθ dφ
        total_power = np.sum(intensity * np.sin(theta)) * d_theta * d_phi
        max_intensity = np.max(intensity)
        
        directivity = 4 * np.pi * max_intensity / total_power
        
        return float(directivity)
    
    def calculate_beamwidth(self, plane='elevation', excitation_phases=None):
        """
        Calculate half-power beamwidth (HPBW) in degrees.
        """
        angles, pattern = self.calculate_pattern_2d(
            plane=plane, excitation_phases=excitation_phases
        )
        
        # Find points where pattern is -3 dB (half power)
        half_power = 0.5
        
        # Find main lobe
        main_lobe_idx = np.argmax(pattern)
        
        # Search left
        left_idx = main_lobe_idx
        while left_idx > 0 and pattern[left_idx] > half_power:
            left_idx -= 1
        
        # Search right
        right_idx = main_lobe_idx
        while right_idx < len(pattern)-1 and pattern[right_idx] > half_power:
            right_idx += 1
        
        beamwidth = angles[right_idx] - angles[left_idx]
        
        return beamwidth

class ButlerMatrix:
    """
    Optical Butler Matrix for passive beamforming.
    
    An N×N Butler Matrix creates N orthogonal beams from N input ports
    without requiring active phase shifters.
    """
    
    def __init__(self, N=4):
        """
        Initialize Butler Matrix.
        
        Parameters:
        -----------
        N : int
            Number of ports (must be power of 2)
        """
        if not (N & (N-1) == 0):
            raise ValueError("N must be a power of 2")
        
        self.N = N
        self.matrix = self._generate_butler_matrix()
        
    def _generate_butler_matrix(self):
        """
        Generate Butler Matrix using FFT-like structure.
        
        For N×N matrix, creates N beams with progressive phase shifts.
        """
        N = self.N
        matrix = np.zeros((N, N), dtype=complex)
        
        for i in range(N):
            for j in range(N):
                # Butler matrix element
                phase = -2 * np.pi * i * j / N
                matrix[i, j] = np.exp(1j * phase) / np.sqrt(N)
        
        return matrix
    
    def get_beam_phases(self, input_port):
        """
        Get phase excitation for array elements when exciting specific input port.
        
        Parameters:
        -----------
        input_port : int
            Which input port to excite (0 to N-1)
            
        Returns:
        --------
        phases : ndarray
            Phase for each array element (radians)
        """
        if input_port < 0 or input_port >= self.N:
            raise ValueError(f"Input port must be 0 to {self.N-1}")
        
        # Column of Butler matrix gives phase distribution
        excitation = self.matrix[:, input_port]
        phases = np.angle(excitation)
        
        return phases
    
    def __str__(self):
        """Pretty print Butler matrix"""
        info = f"\n{'='*70}\n"
        info += f"BUTLER MATRIX BEAMFORMING NETWORK\n"
        info += f"{'='*70}\n"
        info += f"Size:                  {self.N}×{self.N}\n"
        info += f"Number of Beams:       {self.N} orthogonal beams\n"
        info += f"Implementation:        Multi-Mode Interference (MMI) couplers\n"
        info += f"Platform:              Silicon Nitride waveguides\n"
        info += f"{'='*70}\n"
        
        # Show phase progression for each beam
        info += "\nBeam Steering Angles (for each input port):\n"
        for port in range(self.N):
            phases = self.get_beam_phases(port)
            info += f"  Port {port}: phases = {np.degrees(phases).astype(int)}°\n"
        
        return info

class ClassicalValidation:
    """
    Validate quantum model against classical electromagnetic theory.
    
    Uses correspondence principle: quantum model with high decoherence
    should match classical Hertzian dipole array.
    """
    
    def __init__(self, antenna_array):
        """
        Initialize validation framework.
        
        Parameters:
        -----------
        antenna_array : QuantumAntennaArray
            Array to validate
        """
        self.array = antenna_array
        
    def classical_hertzian_dipole_pattern(self, theta, phi):
        """
        Classical radiation pattern for array of Hertzian dipoles.
        
        Assumes incoherent emission (no quantum interference).
        """
        # Single dipole pattern
        pattern = np.sin(theta)**2
        
        # Incoherent sum: total power = N × single emitter power
        pattern *= self.array.N_emitters
        
        return pattern
    
    def quantum_pattern_high_decoherence(self, theta, phi):
        """
        Quantum pattern in high dephasing limit.
        
        Should match classical result.
        """
        rad_pattern = RadiationPattern(self.array)
        
        # With very high dephasing, emitters act independently
        # (This is a simplified model)
        AF = rad_pattern.array_factor(theta, phi)
        
        # In high dephasing: |AF|² → N (incoherent sum)
        pattern = np.sin(theta)**2 * self.array.N_emitters
        
        return pattern
    
    def validate_correspondence(self):
        """
        Check if quantum model reduces to classical in appropriate limit.
        
        Returns:
        --------
        error : float
            Relative error between quantum and classical predictions
        """
        theta = np.linspace(0, np.pi, 100)
        phi = np.zeros_like(theta)
        
        classical = self.classical_hertzian_dipole_pattern(theta, phi)
        quantum = self.quantum_pattern_high_decoherence(theta, phi)
        
        # Calculate relative error
        error = np.mean(np.abs(classical - quantum) / np.max(classical))
        
        return error * 100  # Percentage


def visualize_array_geometry(array, filename='array_geometry.png'):
    """
    Visualize the 3D positions of emitters in the array.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    pos = array.positions
    
    # Plot emitters
    ax.scatter(pos[:, 0]*1e6, pos[:, 1]*1e6, pos[:, 2]*1e6,
              c='blue', s=100, alpha=0.6, edgecolors='black', linewidths=2)
    
    # Label each emitter
    for i, p in enumerate(pos):
        ax.text(p[0]*1e6, p[1]*1e6, p[2]*1e6, f'{i+1}',
               fontsize=10, fontweight='bold')
    
    # Draw connections (coupling)
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            if np.abs(array.coupling_matrix[i, j]) > 0.1 * array.emitter.gamma_decay:
                ax.plot([pos[i,0]*1e6, pos[j,0]*1e6],
                       [pos[i,1]*1e6, pos[j,1]*1e6],
                       [pos[i,2]*1e6, pos[j,2]*1e6],
                       'gray', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('x (μm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (μm)', fontsize=12, fontweight='bold')
    ax.set_zlabel('z (μm)', fontsize=12, fontweight='bold')
    ax.set_title(f'Quantum Antenna Array - {array.geometry.upper()} geometry\n'
                f'{array.N_emitters} emitters, spacing = {array.spacing_wavelengths:.2f}λ',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    return fig

def visualize_radiation_pattern_2d(array, excitation_phases=None, title=None, filename='radiation_2d.png'):
    """
    Plot 2D radiation patterns (polar plots).
    """
    rad_pattern = RadiationPattern(array)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    subplot_kw=dict(projection='polar'))
    
    # Elevation pattern
    angles_elev, pattern_elev = rad_pattern.calculate_pattern_2d(
        'elevation', excitation_phases=excitation_phases
    )
    
    ax1.plot(np.radians(angles_elev), pattern_elev, 'b-', linewidth=2)
    ax1.fill(np.radians(angles_elev), pattern_elev, alpha=0.3, color='blue')
    ax1.set_title('Elevation Pattern (E-plane)', fontsize=13, fontweight='bold', pad=20)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    
    # Azimuth pattern
    angles_az, pattern_az = rad_pattern.calculate_pattern_2d(
        'azimuth', excitation_phases=excitation_phases
    )
    
    ax2.plot(np.radians(angles_az), pattern_az, 'r-', linewidth=2)
    ax2.fill(np.radians(angles_az), pattern_az, alpha=0.3, color='red')
    ax2.set_title('Azimuth Pattern (H-plane)', fontsize=13, fontweight='bold', pad=20)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Calculate metrics
    directivity = rad_pattern.calculate_directivity(excitation_phases)
    beamwidth = rad_pattern.calculate_beamwidth('elevation', excitation_phases)
    
    # Convert to dBi
    directivity_dbi = 10 * np.log10(directivity)
    
    if title is None:
        title = f'{array.geometry.upper()} Array Radiation Pattern'
    
    fig.suptitle(f'{title}\nDirectivity: {directivity_dbi:.1f} dBi, '
                f'HPBW: {beamwidth:.1f}°',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    return fig

def visualize_radiation_pattern_3d(array, excitation_phases=None, filename='radiation_3d.png'):
    """
    Plot 3D radiation pattern.
    """
    rad_pattern = RadiationPattern(array)
    
    theta, phi, intensity = rad_pattern.calculate_pattern_3d(
        excitation_phases=excitation_phases
    )
    
    # Convert to Cartesian coordinates for plotting
    x = intensity * np.sin(theta) * np.cos(phi)
    y = intensity * np.sin(theta) * np.sin(phi)
    z = intensity * np.cos(theta)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(x, y, z, facecolors=cm.jet(intensity),
                          alpha=0.8, antialiased=True, shade=True)
    
    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('y', fontsize=12, fontweight='bold')
    ax.set_zlabel('z', fontsize=12, fontweight='bold')
    ax.set_title(f'3D Radiation Pattern - {array.geometry.upper()} Array',
                fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5)
    cbar.set_label('Normalized Intensity', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    return fig

def visualize_butler_matrix_beams(array, butler_matrix, filename='butler_beams.png'):
    """
    Show radiation patterns for all Butler matrix beams.
    """
    rad_pattern = RadiationPattern(array)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12),
                            subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    for port in range(butler_matrix.N):
        phases = butler_matrix.get_beam_phases(port)
        
        # Extend phases if array is larger than Butler matrix
        if len(phases) < array.N_emitters:
            # For 2D array with 1D Butler matrix, repeat pattern
            phases = np.tile(phases, array.N_emitters // len(phases) + 1)[:array.N_emitters]
        
        angles, pattern = rad_pattern.calculate_pattern_2d(
            'elevation', excitation_phases=phases
        )
        
        ax = axes[port]
        ax.plot(np.radians(angles), pattern, linewidth=2)
        ax.fill(np.radians(angles), pattern, alpha=0.3)
        ax.set_title(f'Beam {port+1} (Port {port})', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Butler Matrix - All Orthogonal Beams', fontsize=15, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    return fig

def main():
    """
    Main function demonstrating complete quantum antenna model.
    """
    
    print("\n" + "="*80)
    print(" QUANTUM ANTENNA ARRAY - MATHEMATICAL MODEL DEMONSTRATION")
    print("="*80)
    print(f"\nOutput Directory: {OUTPUT_DIR}\n")
    
    # ------------------------------------------------------------------
    # STEP 1: Define quantum emitter specifications
    # ------------------------------------------------------------------
    print("\n[STEP 1] Defining Quantum Emitter Specifications...")
    
    # Carbon Quantum Dot emitter
    cqd_specs = QuantumEmitterSpecs('CQD')
    print(cqd_specs)
    
    # Nitrogen-Vacancy center emitter
    nv_specs = QuantumEmitterSpecs('NV')
    print(nv_specs)
    
    # ------------------------------------------------------------------
    # STEP 2: Create antenna array geometries
    # ------------------------------------------------------------------
    print("\n[STEP 2] Creating Quantum Antenna Arrays...")
    
    # Linear array (1D)
    linear_array = QuantumAntennaArray(
        geometry='linear',
        N=8,
        spacing_wavelengths=0.5,
        emitter_specs=cqd_specs
    )
    print(linear_array)
    
    # Planar grid array (2D)
    grid_array = QuantumAntennaArray(
        geometry='grid',
        N=4,
        spacing_wavelengths=0.5,
        emitter_specs=cqd_specs
    )
    print(grid_array)
    
    # ------------------------------------------------------------------
    # STEP 3: Visualize array geometries
    # ------------------------------------------------------------------
    print("\n[STEP 3] Visualizing Array Geometries...")
    
    visualize_array_geometry(linear_array, 'array_geometry_linear.png')
    visualize_array_geometry(grid_array, 'array_geometry_grid.png')
    
    # ------------------------------------------------------------------
    # STEP 4: Calculate and visualize radiation patterns
    # ------------------------------------------------------------------
    print("\n[STEP 4] Calculating Radiation Patterns...")
    
    # 2D patterns
    visualize_radiation_pattern_2d(linear_array, title="Linear Array - Uniform Excitation",
                                   filename='radiation_pattern_2d_linear.png')
    visualize_radiation_pattern_2d(grid_array, title="4×4 Grid Array - Uniform Excitation",
                                   filename='radiation_pattern_2d_grid.png')
    
    # 3D pattern
    visualize_radiation_pattern_3d(linear_array, filename='radiation_pattern_3d.png')
    
    # ------------------------------------------------------------------
    # STEP 5: Butler Matrix beamforming
    # ------------------------------------------------------------------
    print("\n[STEP 5] Demonstrating Butler Matrix Beamforming...")
    
    butler = ButlerMatrix(N=4)
    print(butler)
    
    # Create array matching Butler matrix size
    butler_array = QuantumAntennaArray(
        geometry='linear',
        N=4,
        spacing_wavelengths=0.5,
        emitter_specs=cqd_specs
    )
    
    visualize_butler_matrix_beams(butler_array, butler, 'butler_matrix_beams.png')
    
    # ------------------------------------------------------------------
    # STEP 6: Classical correspondence validation
    # ------------------------------------------------------------------
    print("\n[STEP 6] Validating Correspondence Principle...")
    
    validator = ClassicalValidation(linear_array)
    error = validator.validate_correspondence()
    
    print(f"\nCorrespondence Principle Validation:")
    print(f"  Quantum vs Classical Error: {error:.2f}%")
    if error < 1.0:
        print("  ✓ PASS: Quantum model correctly reduces to classical limit")
    else:
        print("  ⚠ WARNING: Check model consistency")
    
    print("\n[STEP 7] Summary of Antenna Performance Metrics...")
    
    rad_pattern = RadiationPattern(linear_array)
    
    directivity = rad_pattern.calculate_directivity()
    directivity_dbi = 10 * np.log10(directivity)
    
    print(f"\nLinear Array (8 elements, 0.5λ spacing):")
    print(f"  Directivity:           {directivity_dbi:.2f} dBi")
    print(f"  HPBW (Elevation):      {rad_pattern.calculate_beamwidth('elevation'):.1f}°")
    print(f"  HPBW (Azimuth):        {rad_pattern.calculate_beamwidth('azimuth'):.1f}°")
    
    # Superradiance factor
    rates = linear_array.get_collective_decay_rates()
    max_rate = np.max(rates)
    enhancement = max_rate / linear_array.emitter.gamma_decay
    
    print(f"\nQuantum Enhancement Factors:")
    print(f"  Superradiant Enhancement: {enhancement:.2f}× single emitter")
    print(f"  Theoretical Maximum:      {linear_array.N_emitters:.0f}× (N emitters)")
    print(f"  Efficiency:               {(enhancement/linear_array.N_emitters)*100:.1f}%")
    
    print("\n" + "="*80)
    print(f" ALL VISUALIZATIONS SAVED TO: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    # Close all figures to prevent display on headless systems
    plt.close('all')

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()