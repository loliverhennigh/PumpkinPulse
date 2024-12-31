import os
import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import dataclasses
import itertools
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

if __name__ == "__main__":

    # IO parameters
    output_dir = "output_circuit_validation"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Constants
    vacuum_permittivity = 8.854187817e-12
    vacuum_permeability = 1.2566370614e-6
    vacuum_c = float(1.0 / np.sqrt(vacuum_permittivity * vacuum_permeability))

    # Coil parameters
    coil_radius = 0.02
    cable_thickness_r = 0.004
    voltage = 1000 # V
    capacitance = 1e-6 # F

    # Time parameters
    simulation_time = 1e-5  # 2 microseconds

    # Compute the inductance
    inductance = vacuum_permeability * coil_radius * (
        np.log(8 * coil_radius / cable_thickness_r) - 2
    )

    # Add a bit for the extra length of the coil
    inductance *= 1.8

    # RLC circuit differential equation
    def rlc_circuit(t, y):
        """
        Differential equations for the RLC circuit.
        y[0] = q (charge)
        y[1] = I (current)
        """
        q, I = y
        dqdt = I
        dIdt = -q / (inductance * capacitance)  # Assuming R = 0
        return [dqdt, dIdt]

    # Initial conditions
    q_0 = capacitance * voltage  # Initial charge on the capacitor
    I_0 = 0  # Initial current in the coil
    initial_conditions = [q_0, I_0]

    # Time parameters
    time_points = np.linspace(0, simulation_time, 1000)

    # Solve the differential equations
    solution = solve_ivp(
        rlc_circuit, [0, simulation_time], initial_conditions, t_eval=time_points, method="RK45"
    )

    # Extract charge and current
    charge = solution.y[0]
    current = solution.y[1]

    # Compute magnetic field at the center of the coil
    B_center = -(vacuum_permeability * current) / (2 * coil_radius)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(solution.t * 1e6, B_center, label="Magnetic Field at Center of Coil (B)")
    plt.xlabel("Time (µs)")
    plt.ylabel("Magnetic Field (T)")
    plt.title("Magnetic Field at Center of Coil Over Time")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "magnetic_field_center.png"))
    plt.close()

    # Save data for validation
    np.savetxt(
        os.path.join(output_dir, "magnetic_field_center_validated.csv"),
        np.column_stack((solution.t, B_center)),
        delimiter=",",
        header="Time (s), Magnetic Field (T)",
    )

    print(f"Results saved in {output_dir}")
