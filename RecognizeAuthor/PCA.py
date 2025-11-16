import numpy as np
from fontTools.misc.bezierTools import epsilon


class PCA:
    def __init__(self,X:np.array,n_components:int):
        '''
        Args

        :param X: the input data
        :param n_components: the number of components you want to keep

        '''

        def __scale_Data():
            """
                Calculates Z = (X - μ) / σ

                Args:
                    X (np.array): The input data, shape (n_samples, n_features)
                Returns:
                    Z (np.array): The standardized data.
                    means (np.array): The mean of each feature (μ).
                    std_devs (np.array): The standard deviation of each feature (σ).
                """
            means = np.mean(X, axis=0)
            std_devs = np.std(X, axis=0)
            epsilon = 1e-10
            Z = (X - means) / (std_devs+epsilon)

            return Z, means,std_devs

        def fit_data():
            """
                 Args:
            X (np.array): The input data, MUST be shape (n_samples, n_features).
                          For STFT, this means you must pass S_db.T.
            n_components (int): The target number of dimensions (k).

            Returns:
                Y_pca (np.array): The final, "flattened" data, shape (n_samples, k).
                W (np.array): The projection matrix, shape (n_features, k).
                means (np.array): The original mean of each feature (μ).
                std_devs (np.array): The original std. dev. of each feature (σ).
                eigenvalues_sorted (np.array): The sorted eigenvalues (λ).
            """

            Z,means,std_devs = __scale_Data()
            n_samples = Z.shape[0]
            # Compute Covariance as the matrix product since mean is 0

            Covariance = 1/ (n_samples - 1) * (Z.T @ Z)

            eigenvalues, eigenvectors = np.linalg.eig(Covariance)
            eigenvectors = np.real(eigenvectors)

            eigenvalues = np.real(eigenvalues)
            # Get highest eigenValues as they contain the most variance
            idx = np.argsort(eigenvalues)[::-1]

            eigenvalues_sorted = eigenvalues[idx]

            # Sort collumns taking all rows in the index order
            eigenvectors_sorted = eigenvectors[:,idx]

            Projection_Matrix = eigenvectors_sorted[:,:n_components]

            Y_PCA = Z@Projection_Matrix



