# Simple analytic model of capacitor discharge into coil

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Function to calculate the inductance of a coil
def coil_inductance(nr_turns, coil_radius, coil_length, mu_0=4 * np.pi * 1e-7):
    """Calculate the inductance of the coil."""
    area = np.pi * coil_radius**2
    return mu_0 * nr_turns**2 * area / coil_length

# Function to calculate the magnetic field in the center of the coil
def magnetic_field(nr_turns, coil_length, current, mu_0=4 * np.pi * 1e-7):
    """Calculate the magnetic field in the center of the coil."""
    return nr_turns * current / coil_length

# Define the ODE for the LC circuit
def lc_circuit(t, y, L, C):
    """ODE for the LC circuit."""
    I, dI_dt = y
    return [dI_dt, -I / (L * C)]

# Constants
mu_0 = 4 * np.pi * 1e-7

# Coil parameters
coil_radius = 0.4  # meters
cable_radius = 0.01  # meters
nr_turns = 10
length = nr_turns * cable_radius * 2  # meters

# Capacitor parameters
voltage = 1.0e3  # Volts
capacitance = 1.0e-3  # Farad

# Calculate inductance
inductance = coil_inductance(nr_turns, coil_radius, length)

# Initial conditions for the LC circuit
initial_current = 0.0  # A
initial_voltage = voltage  # V
initial_current_rate = initial_voltage / inductance  # dI/dt at t=0

# Solve the ODE
t_span = (0, 1e-6)  # Time span for one period
t_eval = np.linspace(t_span[0], t_span[1], 1000)
initial_conditions = [initial_current, initial_current_rate]
solution = solve_ivp(
    lc_circuit,
    t_span,
    initial_conditions,
    args=(inductance, capacitance),
    t_eval=t_eval,
    method="RK45",
)

# Extract current and calculate magnetic field
current = solution.y[0]
magnetic_field_values = magnetic_field(nr_turns, length, current)

# Plot the magnetic field over time
plt.figure(figsize=(10, 6))
plt.plot(solution.t, magnetic_field_values, label="Magnetic Field $B(t)$")
plt.xlabel("Time (s)")
plt.ylabel("Magnetic Field (T)")
plt.title("Magnetic Field Over Time")
plt.grid()
plt.legend()
plt.show()
