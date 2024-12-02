# Performs convergence analysis

from typing import List, Tuple
import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt


def convergence_analysis(
    run_simulation: callable,
    factors: List[int],
    output_dir: str,
):
    """
    Performs convergence analysis for the given generator functions
    """

    # List of errors for each generator
    errors = []

    # Iterate over each generator
    for i, f in enumerate(factors[:-1]): # Use last factor as reference

        # Get downsampling factor
        downsampling_factor = 2 ** (len(factors) - i - 1)

        # Errors for each field
        timestamp_error = {}

        # Iterate over each field
        for lr_fields, hr_fields in zip(run_simulation(f), run_simulation(factors[-1])):

            # Compute error for each field
            for name in lr_fields.keys():
               
                # Make list to store errors for each field
                if name not in timestamp_error:
                    timestamp_error[name] = []

                # Sum of errors
                e = 0

                # for each channel
                for c in range(lr_fields[name].shape[0]):

                    # low-fidelity field
                    lr_field = lr_fields[name][c]

                    # high-fidelity field
                    hr_field = hr_fields[name][c]

                    # downsample high-fidelity field
                    if len(lr_field.shape) == 2:
                        down_sampled_hr_field = hr_field.reshape(
                            hr_field.shape[0] // downsampling_factor,
                            downsampling_factor,
                            hr_field.shape[1] // downsampling_factor,
                            downsampling_factor,
                        ).mean(axis=(1, 3))
                    else:
                        down_sampled_hr_field = hr_field.reshape(
                            hr_field.shape[0] // downsampling_factor,
                            downsampling_factor,
                            hr_field.shape[1] // downsampling_factor,
                            downsampling_factor,
                            hr_field.shape[2] // downsampling_factor,
                            downsampling_factor,
                        ).mean(axis=(1, 3, 5))

                    # Plot lr and downsampled hr side by side and comparison
                    plt.figure()
                    plt.subplot(1, 3, 1)
                    plt.imshow(lr_field, cmap="gray")
                    plt.title("Low-fidelity")
                    plt.subplot(1, 3, 2)
                    plt.imshow(down_sampled_hr_field, cmap="gray")
                    plt.title("Downsampled High-fidelity")
                    plt.subplot(1, 3, 3)
                    plt.imshow(lr_field - down_sampled_hr_field, cmap="gray")
                    plt.title("Difference")
                    plt.savefig(f"{output_dir}/{name}_{c}_{i}.png")

                    # Compute error
                    e += np.sqrt(np.mean((lr_field - down_sampled_hr_field) ** 2))

                # Append error
                timestamp_error[name].append(e / lr_fields[name].shape[0])


        # Compute average error
        errors.append({name: np.mean(timestamp_error[name]) for name in timestamp_error.keys()})

    # Make p plot for each field
    for name in errors[0].keys():

        # Get h array
        h_array = np.array([1.0 / 2 ** i for i in range(len(errors))])

        # Get error array
        error_array = np.array([error[name] for error in errors])

        # Log-log plot
        log_h_array = np.log(h_array)
        log_error_array = np.log(error_array)

        # Fit line
        p, c = np.polyfit(log_h_array, log_error_array, 1)

        # Make plot
        plt.figure()
        plt.loglog(h_array, error_array, "o-", label="Error")
        plt.loglog(h_array, np.exp(c) * h_array ** p, "--", label=f"Fit: p={p:.2f}")
        plt.xlabel("h")
        plt.ylabel("Error")
        plt.title(f"Convergence for {name}, Convergence rate: {p:.2f}")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(f"{output_dir}/{name}_convergence.png")
