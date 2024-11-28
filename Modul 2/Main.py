import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Estimation import Estimation
import Kalmanfilter as kf

# Define the base data folder
script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir,'Data')
output_folder = os.path.join(script_dir,'./Output')
os.makedirs(output_folder, exist_ok=True)  # Ensure Output folder exists

# Initialize estimation object with all necessary files
calibration_file = os.path.join(data_folder, '0-calibration_fts-accel.csv')
accel_file = os.path.join(data_folder, '0-steady-state_accel.csv')
fts_file = os.path.join(data_folder, '0-steady-state_wrench.csv')

# Pass all three files to the Estimation class
estimation = Estimation(calibration_file, accel_file, fts_file)

# Perform bias, mass estimation, and variance calculations
estimation.calculate_biases()
estimation.estimate_mass_and_force_bias()
estimation.estimate_mass_center_and_torque_bias()
estimation.calculate_variances()
results = estimation.get_results()

# Extract biases, estimated values, and variances
force_bias = np.array(results['force_bias'])
torque_bias = np.array(results['torque_bias'])
imu_bias = np.array(results['imu_bias'])
mass = results['mass_est']
mass_center = np.array(results['mass_center'])
variances = results['variances']

# Save estimation results to a CSV file
estimation_results = pd.DataFrame({
    'Parameter': [
        'Force Bias X', 'Force Bias Y', 'Force Bias Z',
        'Torque Bias X', 'Torque Bias Y', 'Torque Bias Z',
        'IMU Bias X', 'IMU Bias Y', 'IMU Bias Z',
        'Mass', 'Mass Center X', 'Mass Center Y', 'Mass Center Z',
        'Ax Variance', 'Ay Variance', 'Az Variance',
        'Fx Variance', 'Fy Variance', 'Fz Variance',
        'Tx Variance', 'Ty Variance', 'Tz Variance'
    ],
    'Value': [
        *force_bias.tolist(),
        *torque_bias.tolist(),
        *imu_bias.tolist(),
        mass,
        *mass_center.tolist(),
        variances['ax_variance'], variances['ay_variance'], variances['az_variance'],
        variances['fx_variance'], variances['fy_variance'], variances['fz_variance'],
        variances['tx_variance'], variances['ty_variance'], variances['tz_variance']
    ]
})
estimation_results_file = os.path.join(output_folder, "estimation_results.csv")
estimation_results.to_csv(estimation_results_file, index=False)
print(f"Estimation results saved to {estimation_results_file}")

# Gravitational constants
gravity_vector_world = np.array([0, 0, 9.81])  # For force calculations

# Function to align datasets based on their timestamps
def align_data(accel_data, force_data, orientation_data, accel_rate, force_rate, orientation_rate):
    accel_data['t'] /= 1e6
    force_data['t'] /= 1e6
    orientation_data['t'] /= 1e6
    accel_data.sort_values(by='t', inplace=True)
    force_data.sort_values(by='t', inplace=True)
    orientation_data.sort_values(by='t', inplace=True)
    merged = pd.merge_asof(accel_data, force_data, on='t', direction='nearest', tolerance=1 / force_rate)
    merged = pd.merge_asof(merged, orientation_data, on='t', direction='nearest', tolerance=1 / orientation_rate)
    merged = merged.ffill()
    return merged

# Function to calculate z_c_hat
def calculate_zc(state_estimate, mass_s, r_s):
    I3 = np.eye(3)
    zeros_3x3 = np.zeros((3, 3))
    r_s_cross = np.array([
        [0, -r_s[2], r_s[1]],
        [r_s[2], 0, -r_s[0]],
        [-r_s[1], r_s[0], 0]
    ])
    M = np.vstack((
        np.hstack((-mass_s * I3, I3, zeros_3x3)),
        np.hstack((-mass_s * r_s_cross, zeros_3x3, I3))
    ))
    return M @ state_estimate

# Function to stabilize data by removing the initial noisy part
def stabilize_data(times, F3, T2, x6, x8, zc3, zc5, stabilization_period=10):
    return (
        times[stabilization_period:],
        F3[stabilization_period:],
        T2[stabilization_period:],
        x6[stabilization_period:],
        x8[stabilization_period:],
        zc3[stabilization_period:],
        zc5[stabilization_period:]
    )

# Function to process data using the Kalman Filter
def filter_data(dataset):
    timestamps, forces, torques = [], [], []
    x_estimates = []
    zc_estimates = []

    state_size = 9
    measurement_size = 9
    transition_matrix = np.eye(state_size)
    measurement_matrix = np.zeros((measurement_size, state_size))
    measurement_matrix[:3, :3] = np.eye(3)
    measurement_matrix[3:, 3:] = np.eye(6)
    process_noise_cov = np.eye(state_size) * 0.07
    force_measurement_scaling = 0.753
    torque_measurement_scaling = 50

    imu_variances = np.array([variances['ax_variance'], variances['ay_variance'], variances['az_variance']])
    force_variances = np.array([variances['fx_variance'], variances['fy_variance'], variances['fz_variance']]) * force_measurement_scaling
    torque_variances = np.array([variances['tx_variance'], variances['ty_variance'], variances['tz_variance']]) * torque_measurement_scaling

    measurement_variances = np.hstack([imu_variances, force_variances, torque_variances])
    R_measurement = np.diag(measurement_variances)

    initial_covariance = np.eye(state_size)
    initial_state = np.zeros((state_size, 1))

    kalman_filter = kf.Estimator(transition_matrix, measurement_matrix, process_noise_cov, R_measurement, initial_covariance, initial_state)

    min_time = dataset['t'].min()
    dataset['t'] -= min_time

    for _, row in dataset.iterrows():
        timestamps.append(row['t'])
        wrench_values = row[['fx', 'fy', 'fz', 'tx', 'ty', 'tz']].values
        accel_values = row[['ax', 'ay', 'az']].values

        wrench_values[:3] -= force_bias
        wrench_values[3:] -= torque_bias
        accel_values -= imu_bias

        rotation_matrix = np.array([
            [row['r11'], row['r12'], row['r13']],
            [row['r21'], row['r22'], row['r23']],
            [row['r31'], row['r32'], row['r33']],
        ])
        gravity_sensor = rotation_matrix.T @ gravity_vector_world
        accel_corrected = accel_values - gravity_sensor

        z_observation = np.hstack((accel_corrected, wrench_values))

        kalman_filter.predict()
        kalman_filter.update(z_observation)

        zc_hat = calculate_zc(kalman_filter.state, mass, mass_center)
        x_estimates.append(kalman_filter.state.copy())
        zc_estimates.append(zc_hat.copy())

        forces.append(wrench_values[2])
        torques.append(wrench_values[4])

    return (
        np.array(timestamps),
        np.array(forces),
        np.array(torques),
        np.array(x_estimates),
        np.array(zc_estimates)
    )

# Define experiment files and their corresponding conditions
test_conditions = [
    ('1-baseline_accel.csv', '1-baseline_wrench.csv', '1-baseline_orientations.csv', 'Undisturbed'),
    ('2-vibrations_accel.csv', '2-vibrations_wrench.csv', '2-vibrations_orientations.csv', 'Vibrations'),
    ('3-vibrations-contact_accel.csv', '3-vibrations-contact_wrench.csv', '3-vibrations-contact_orientations.csv', 'Vibrations + Contact')
]

plt.figure(figsize=(19, 10))

for idx, (accel_file, force_file, orientation_file, label) in enumerate(test_conditions):
    accel_file_path = os.path.join(data_folder, accel_file)
    force_file_path = os.path.join(data_folder, force_file)
    orientation_file_path = os.path.join(data_folder, orientation_file)

    accel_data = pd.read_csv(accel_file_path)
    force_data = pd.read_csv(force_file_path)
    orientation_data = pd.read_csv(orientation_file_path)

    combined_data = align_data(accel_data, force_data, orientation_data, accel_rate=254.3, force_rate=698.3, orientation_rate=100.2)

    if combined_data.empty:
        print(f"Skipping {label} due to insufficient data.")
        continue

    times, F3, T2, x_estimates, zc_estimates = filter_data(combined_data)

    x6 = np.array([state[5, 0] for state in x_estimates])  # Estimated force
    x8 = np.array([state[7, 0] for state in x_estimates])  # Estimated torque
    zc3 = np.array([zc[2, 0] for zc in zc_estimates])      # Contact force
    zc5 = np.array([zc[4, 0] for zc in zc_estimates])      # Contact torque

    # Stabilize data
    times, F3, T2, x6, x8, zc3, zc5 = stabilize_data(times, F3, T2, x6, x8, zc3, zc5)

    # Save adjusted signals
    adjusted_signals_file = os.path.join(output_folder, f"adjusted_signals_{label.replace(' ', '_').lower()}.csv")
    adjusted_signals = pd.DataFrame({'timestamp': times, 'F3': F3, 'T2': T2})
    adjusted_signals.to_csv(adjusted_signals_file, index=False)

    # Ensure arrays have matching lengths for state vector estimates
    min_length = min(len(times), len(x_estimates))
    times = times[:min_length]
    x_estimates = x_estimates[:min_length]

    # Save state vector estimates
    state_vector_file = os.path.join(output_folder, f"state_vector_{label.replace(' ', '_').lower()}.csv")
    state_estimates_df = pd.DataFrame({
        'timestamp': times,
        **{f'x{i}': [state[i, 0] for state in x_estimates] for i in range(9)}
    })
    state_estimates_df.to_csv(state_vector_file, index=False)

    # Save contact wrench estimates
    contact_wrench_file = os.path.join(output_folder, f"contact_wrench_estimates_{label.replace(' ', '_').lower()}.csv")
    zc_estimates_df = pd.DataFrame({
        'timestamp': times,
        **{f'zc{i}': [zc[i, 0] for zc in zc_estimates[:len(times)]] for i in range(zc_estimates[0].shape[0])}
    })
    zc_estimates_df.to_csv(contact_wrench_file, index=False)

    print(f"Data saved for {label}:\n  - Adjusted signals: {adjusted_signals_file}\n  - State vector: {state_vector_file}\n  - Contact wrench: {contact_wrench_file}")

    # Plot force data
    plt.subplot(3, 2, idx * 2 + 1)
    plt.plot(times, F3, label=r'$F_3$', color='blue')
    plt.plot(times, x6, label=r'$\hat{X}_6$', color='orange')
    plt.plot(times, zc3, label=r'$\hat{Z}_{c,3}$', color='green')
    plt.xlabel('Time')
    plt.ylabel('Force')
    plt.legend()

    # Plot torque data
    plt.subplot(3, 2, idx * 2 + 2)
    plt.plot(times, T2, label=r'$T_2$', color='blue')
    plt.plot(times, x8, label=r'$\hat{X}_8$', color='orange')
    plt.plot(times, zc5, label=r'$\hat{Z}_{c,5}$', color='green')
    plt.xlabel('Time')
    plt.ylabel('Torque')
    plt.legend()

plt.subplots_adjust(top=0.962,bottom=0.06,left=0.045,right=0.992,hspace=0.337,wspace=0.1)
plt.tight_layout()
plt.show()