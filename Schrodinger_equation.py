import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh  # For solving eigenvalues of symmetric matrices

# Constants
hbar = 1.0  # Reduced Planck's constant (set to 1 in natural units)
m = 1.0     # Mass of the particle (set to 1 in natural units)
L = 1.0     # Length of the potential well
N = 100     # Number of grid points
h = L / (N + 1)  # Grid spacing

# Finite difference coefficient
C = hbar**2 / (2 * m * h**2)

# Construct the Hamiltonian matrix
H = np.zeros((N, N))

for i in range(N):
    H[i, i] = 2 * C   
    if i > 0:
        H[i, i-1] = -C  
    if i < N - 1:
        H[i, i+1] = -C  

# Solve for eigenvalues and eigenvectors
eigenvalues, eigenvectors = eigh(H)

# Select first few eigenstates for plotting
num_states = 3  # Number of states to visualize

# Plot eigenfunctions
x = np.linspace(0, L, N+2)  # boundary points
plt.figure(figsize=(8, 6))

for n in range(num_states):
    psi_n = np.zeros(N+2)
    psi_n[1:N+1] = eigenvectors[:, n]  # Insert the eigenvector inside the well
    psi_n /= np.sqrt(np.trapz(psi_n**2, x))  

    plt.plot(x, psi_n, label=f"n={n+1}, E={eigenvalues[n]:.3f}")

plt.xlabel("Position (x)")
plt.ylabel("Wavefunction ψ(x)")
plt.title("Wavefunctions for the Infinite Square Well")
plt.legend()
plt.show()

# Print eigenvalues
print("First few energy levels:")
for i in range(num_states):
    print(f"E_{i+1} = {eigenvalues[i]:.5f}")
