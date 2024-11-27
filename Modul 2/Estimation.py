import pandas as pd
import numpy as np

class Estimation:
    def __init__(self, calibration_file_path, accel_file_path, fts_file_path):
        # Load data from provided file paths
        self.calibration_data = pd.read_csv(calibration_file_path)
        self.accel_data = pd.read_csv(accel_file_path) * 9.81
        self.fts_data = pd.read_csv(fts_file_path)
        
        # Initialize placeholders for results
        self.force_bias = None
        self.torque_bias = None
        self.imu_bias = None
        self.mass_est = None
        self.mass_center = None
        self.variances = {}

    def calculate_biases(self):
        # Calculate force bias from calibration data
        self.force_bias = self.calibration_data[['fx', 'fy', 'fz']].mean().values.tolist()

        # Calculate IMU bias from calibration data
        self.imu_bias = self.calibration_data[['ax', 'ay', 'az']].mean().values.tolist()

        # Calculate torque bias from calibration data
        self.torque_bias = [
            self.calibration_data['tx'][:8].mean(),  # First 8 measurements
            self.calibration_data['ty'][8:16].mean(),  # Next 8 measurements
            self.calibration_data['tz'][16:24].mean()  # Last 8 measurements
        ]

    def estimate_mass_and_force_bias(self):
        # Prepare the data for least squares estimation
        Fx, Fy, Fz = self.calibration_data['fx'].values, self.calibration_data['fy'].values, self.calibration_data['fz'].values
        gsx, gsy, gsz = self.calibration_data['gx'].values, self.calibration_data['gy'].values, self.calibration_data['gz'].values

        N = len(Fx)
        aForce = np.zeros((3 * N, 4))  # Initialize matrix with correct dimensions
        bForce = np.zeros(3 * N)

        # Construct aForce matrix and bForce vector
        for i in range(N):
            aForce[3 * i:3 * i + 3, :] = [
                [1, 0, 0, gsx[i]],
                [0, 1, 0, gsy[i]],
                [0, 0, 1, gsz[i]]
            ]
            bForce[3 * i:3 * i + 3] = [Fx[i], Fy[i], Fz[i]]

        # Solve least squares problem to find force bias and estimated mass
        xForce, _, _, _ = np.linalg.lstsq(aForce, bForce, rcond=None)

        # Store estimated force bias and mass
        self.force_bias = xForce[:3].tolist()
        self.mass_est = xForce[3]

    def estimate_mass_center_and_torque_bias(self):
        Tx, Ty, Tz = self.calibration_data['tx'].values, self.calibration_data['ty'].values, self.calibration_data['tz'].values
        gsx, gsy, gsz = self.calibration_data['gx'].values, self.calibration_data['gy'].values, self.calibration_data['gz'].values

        N = len(Tx)
        aTorque = np.zeros((3 * N, 6))  # Initialize matrix with correct dimensions
        bTorque = np.zeros(3 * N)

        # Construct aTorque matrix and bTorque vector
        for i in range(N):
            Ai = np.array([[0, -gsz[i], gsy[i]],
                           [gsz[i], 0, -gsx[i]],
                           [-gsy[i], gsx[i], 0]])

            aTorque[3 * i:3 * i + 3, :] = [
                [1, 0, 0, self.mass_est * Ai[0, 0], self.mass_est * Ai[0, 1], self.mass_est * Ai[0, 2]],
                [0, 1, 0, self.mass_est * Ai[1, 0], self.mass_est * Ai[1, 1], self.mass_est * Ai[1, 2]],
                [0, 0, 1, self.mass_est * Ai[2, 0], self.mass_est * Ai[2, 1], self.mass_est * Ai[2, 2]]
            ]
            bTorque[3 * i:3 * i + 3] = [Tx[i], Ty[i], Tz[i]]

        # Solve least squares problem to find torque bias and estimated mass center
        xTorque, _, _, _ = np.linalg.lstsq(aTorque, bTorque, rcond=None)

        # Store estimated torque bias and mass center
        self.torque_bias = xTorque[:3].tolist()
        self.mass_center = np.abs(xTorque[3:6].tolist())

    def calculate_variances(self):
        # Define rotation matrix R_fa
        R_fa = np.array([[0, -1, 0],
                         [0, 0, 1],
                         [-1, 0, 0]])

        # Apply rotation matrix to IMU data
        imu_rotated = self.accel_data[['ax', 'ay', 'az']].dot(R_fa.T)

        # Calculate variances for rotated IMU data (scaled)
        self.variances['ax_variance'] = imu_rotated.iloc[:, 0].var() * 100
        self.variances['ay_variance'] = imu_rotated.iloc[:, 1].var() * 100
        self.variances['az_variance'] = imu_rotated.iloc[:, 2].var() * 100

        # Calculate variances for FTS force data (scaled)
        self.variances['fx_variance'] = self.fts_data['fx'].var() * 250
        self.variances['fy_variance'] = self.fts_data['fy'].var() * 250
        self.variances['fz_variance'] = self.fts_data['fz'].var() * 250

        # Calculate variances for FTS torque data (scaled)
        self.variances['tx_variance'] = self.fts_data['tx'].var() * 5000
        self.variances['ty_variance'] = self.fts_data['ty'].var() * 5000
        self.variances['tz_variance'] = self.fts_data['tz'].var() * 5000

    def get_results(self):
        return {
            "force_bias": self.force_bias,
            "torque_bias": self.torque_bias,
            "imu_bias": self.imu_bias,
            "mass_est": self.mass_est,
            "mass_center": self.mass_center,
            "variances": self.variances
        }