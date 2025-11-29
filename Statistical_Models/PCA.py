# (REPLACE THIS FILE)
# FILE: PCA.py

import numpy as np


class PCA:
    """
    A simple PCA implementation with a sklearn-style
    fit/transform interface.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.projection_matrix = None
        self.means = None
        self.std_devs = None
        self.epsilon = 1e-10  # For numerical stability

    def _scale_data(self, X: np.array) -> np.array:
        """
        Scales the data using the *stored* means and std_devs.
        If they don't exist (i.e., during fit), it calculates and stores them.
        """
        if self.means is None or self.std_devs is None:
            self.means = np.mean(X, axis=0)
            self.std_devs = np.std(X, axis=0)

        # Standardize the data: Z = (X - μ) / (σ + ε)
        Z = (X - self.means) / (self.std_devs + self.epsilon)
        return Z

    def fit(self, X: np.array):
        """
        Fits the PCA model to the data X.

        Args:
            X (np.array): Input data, shape (n_samples, n_features).
                          For STFT, this should be (n_frames, n_bins).
        """
        # 1. Standardize the data
        Z = self._scale_data(X)
        n_samples = Z.shape[0]

        # 2. Compute Covariance Matrix
        # Since mean is 0, Cov = 1/(n-1) * (Z.T @ Z)
        Covariance = (1.0 / (n_samples - 1)) * (Z.T @ Z)

        # 3. Get Eigenvectors and Eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(Covariance)

        # 4. Sort eigenvectors by eigenvalues
        # Ensure we are working with real numbers (eig can return complex)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors_sorted = eigenvectors[:, idx]

        # 5. Store the projection matrix (the first k eigenvectors)
        self.projection_matrix = eigenvectors_sorted[:, :self.n_components]

    def transform(self, X: np.array) -> np.array:
        """
        Transforms new data X using the fitted PCA model.

        Args:
            X (np.array): New data, shape (n_new_samples, n_features).

        Returns:
            Y_pca (np.array): Transformed data, shape (n_new_samples, n_components).
        """
        if self.projection_matrix is None:
            raise RuntimeError("PCA must be fitted before transforming data.")

        # 1. Scale the new data using the *original* means and std_devs
        Z = self._scale_data(X)

        # 2. Project onto the new component space
        Y_pca = Z @ self.projection_matrix
        return Y_pca