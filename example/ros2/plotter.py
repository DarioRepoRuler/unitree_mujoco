import matplotlib.pyplot as plt
import re


def plot_foot_force(filename):
    forces_by_index = [[], [], [], []]  # Separate lists for each index within a timestep

    # Read the file
    with open(filename, 'r') as file:
        for line in file:
            # Check if the line contains foot force data
            if line.startswith("foot force:"):
                # Extract numeric values from the comma-separated line
                values = line.split(':')[1].strip().split(',')
                values = [int(value) for value in values if value]
                # Append values to respective index lists
                for i, value in enumerate(values):
                    forces_by_index[i].append(value)

    # Plot each series separately
    plt.figure(figsize=(10, 5))
    for i, series in enumerate(forces_by_index):
        plt.plot(series, marker='o', linestyle='-', label=f'Foot Force {i + 1}')
    
    plt.title("Foot Force Graphs")
    plt.xlabel("Timestep Index")
    plt.ylabel("Force Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Usage example
plot_foot_force('foot_force_log_vic.csv')