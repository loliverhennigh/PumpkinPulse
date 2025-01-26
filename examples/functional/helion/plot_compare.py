import matplotlib.pyplot as plt
import numpy as np

# FDTD data
fdtd_file = "./output_circuit_validation/magnetic_strength_center_fdtd.csv"
validation_file = "./output_circuit_validation/magnetic_field_center_validated.csv"

# Read data
fdtd_data = np.genfromtxt(fdtd_file, delimiter=',', skip_header=1)
validation_data = np.genfromtxt(validation_file, delimiter=',', skip_header=1)

# Plot for each time step
for i in range(fdtd_data.shape[0]):
    fig = plt.figure( figsize=(16, 9) )
    plt.title('Magnetic field strength at center of the circuit')
    plt.plot(fdtd_data[:i, 0], fdtd_data[:i, 1], label='FDTD')
    plt.plot(validation_data[:, 0], validation_data[:, 1], label='Analytical')

    # Plot dots for the current time step
    plt.plot(fdtd_data[i, 0], fdtd_data[i, 1], 'ro')

    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetic field strength (T)')
    plt.legend()
    plt.savefig(f'./output_circuit_validation/compare_{str(i).zfill(7)}.png')
    plt.close()

