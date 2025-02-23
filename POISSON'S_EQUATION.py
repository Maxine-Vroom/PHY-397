import numpy as np
import matplotlib.pyplot as plt

# Set grid parameters
N = 50   # Number of points in each direction (NxN grid)
L = 1.0  # Physical size of the domain
h = L / (N - 1)  # Grid spacing

# the charge distribution (rho)
rho = np.zeros((N, N))  
rho[N//2, N//2] = 1.0  # Place a point charge at the center

#the potential grid (phi)
phi = np.zeros((N, N))

# Set iteration parameters
max_iterations = 1000
tolerance = 1e-5

# Jacobi 
for iteration in range(max_iterations):
    phi_new = np.copy(phi)  # Copy current potential values

    # Update the grid using the Jacobi method
    for i in range(1, N-1):
        for j in range(1, N-1):
            phi_new[i, j] = 0.25 * (
                phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] - h**2 * rho[i, j]
            )

    # maximum error between successive iterations
    error = np.max(np.abs(phi_new - phi))

    # Update potential grid
    phi = phi_new

    # Check if the solution has converged
    if error < tolerance:
        print(f"Converged after {iteration} iterations")
        break

plt.imshow(phi, extent=[0, L, 0, L], origin='lower', cmap='inferno')
plt.colorbar(label="Potential φ")
plt.title("Numerical Solution of Poisson's Equation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
