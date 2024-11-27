import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Define a class for Kalman Filter implementation
class Estimator:
    def __init__(self, state_transition, measurement_model, process_noise,
                 measurement_noise, covariance, initial_state):
        self.A = state_transition        # State transition matrix
        self.H = measurement_model       # Measurement model matrix
        self.Q = process_noise           # Process noise covariance
        self.R = measurement_noise       # Measurement noise covariance
        self.P = covariance              # Error covariance matrix
        self.state = initial_state       # Initial state vector

    def predict(self):
        # Update state estimate using the process model
        self.state = self.A @ self.state  # No control input
        # Update error covariance
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.state

    def update(self, observation):
        # Compute Kalman gain
        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        kalman_gain = self.P @ self.H.T @ np.linalg.inv(innovation_covariance)
        # Update state estimate using the observation
        innovation = observation.reshape(-1, 1) - self.H @ self.state
        self.state = self.state + kalman_gain @ innovation
        # Update error covariance
        self.P = (np.eye(self.P.shape[0]) - kalman_gain @ self.H) @ self.P
        return self.state