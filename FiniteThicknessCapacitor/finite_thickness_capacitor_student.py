#!/usr/bin/env python3
"""
Module: Finite Thickness Parallel Plate Capacitor (Student Version)
"""

import numpy as np 
import matplotlib.pyplot as plt

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-6):
    V = np.zeros((ny, nx))

    # 设置上下平行板电势
    mid_y = ny // 2
    plate_y1 = mid_y - plate_separation // 2
    plate_y2 = mid_y + plate_separation // 2

    plate_x_start = (nx - plate_thickness) // 2
    plate_x_end = plate_x_start + plate_thickness

    # 上板为1V
    V[plate_y1, plate_x_start:plate_x_end] = 1.0
    # 下板为0V
    V[plate_y2, plate_x_start:plate_x_end] = 0.0

    # 固定边界
    fixed = np.zeros_like(V, dtype=bool)
    fixed[plate_y1, plate_x_start:plate_x_end] = True
    fixed[plate_y2, plate_x_start:plate_x_end] = True

    for iteration in range(max_iter):
        V_old = V.copy()
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                if not fixed[j, i]:
                    V[j, i] = (1 - omega) * V[j, i] + omega * 0.25 * (
                        V[j+1, i] + V[j-1, i] + V[j, i+1] + V[j, i-1]
                    )
        error = np.max(np.abs(V - V_old))
        if error < tolerance:
            print(f"Converged in {iteration} iterations.")
            break
    else:
        print("Warning: Maximum iterations reached.")

    return V

def calculate_charge_density(potential_grid, dx, dy):
    # 使用泊松方程的离散形式估算电荷密度：ρ = -ε₀ (∂²V/∂x² + ∂²V/∂y²)
    epsilon_0 = 8.854e-12
    rho = np.zeros_like(potential_grid)

    rho[1:-1, 1:-1] = -epsilon_0 * (
        (potential_grid[1:-1, 2:] - 2 * potential_grid[1:-1, 1:-1] + potential_grid[1:-1, 0:-2]) / dx**2 +
        (potential_grid[2:, 1:-1] - 2 * potential_grid[1:-1, 1:-1] + potential_grid[0:-2, 1:-1]) / dy**2
    )

    return rho

def plot_results(potential, charge_density, x_coords, y_coords):
    X, Y = np.meshgrid(x_coords, y_coords)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    cp = plt.contourf(X, Y, potential, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Potential (V)')
    plt.title("Electric Potential")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(1, 2, 2)
    cd = plt.contourf(X, Y, charge_density, levels=50, cmap='coolwarm')
    plt.colorbar(cd, label='Charge Density (C/m²)')
    plt.title("Charge Density")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 网格和物理参数
    nx, ny = 100, 100
    plate_thickness = 40
    plate_separation = 20
    dx = dy = 1e-3  # 1 mm 网格间距

    potential = solve_laplace_sor(nx, ny, plate_thickness, plate_separation)
    charge_density = calculate_charge_density(potential, dx, dy)

    x_coords = np.linspace(0, dx * (nx - 1), nx)
    y_coords = np.linspace(0, dy * (ny - 1), ny)

    plot_results(potential, charge_density, x_coords, y_coords)
