import numpy as np
from data_loader import *


class GaussianMixture:
    def __init__(self, n_components, n_iter=100, tol=1e-3):
        self.weights = None
        self.covariances = None
        self.means = None
        self.n_components = n_components  # Number of Gaussian components
        self.n_iter = n_iter  # Maximum number of iterations
        self.tol = tol  # Tolerance for convergence

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize parameters using K-means
        self.means = self._initialize_means_with_kmeans(X)
        self.covariances = np.array([np.eye(n_features)] * self.n_components)
        self.weights = np.ones(self.n_components) / self.n_components

        log_likelihood_old = 0
        for _ in range(self.n_iter):
            # E-step: calculate responsibilities
            responsibilities = self._e_step(X)

            # M-step: update parameters based on responsibilities
            self._m_step(X, responsibilities)

            # Check for convergence with additional safeguard against invalid values
            log_likelihood_new = self._log_likelihood(X)
            if np.isnan(log_likelihood_new) or np.isinf(log_likelihood_new):
                break
            if abs(log_likelihood_new - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood_new

    def _initialize_means_with_kmeans(self, X):
        # Implement K-means to initialize means
        # This is a simple implementation of K-means algorithm
        centroids = X[np.random.choice(X.shape[0], self.n_components, replace=False)]
        for _ in range(10):  # Number of iterations for K-means
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
            centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_components)])
        return centroids

    def _e_step(self, X):
        # Calculate likelihood of each component for each sample
        likelihood = np.array([self._pdf(X, k) for k in range(self.n_components)]).T
        # Calculate responsibilities
        epsilon = 1e-6  # A small constant for numerical stability
        total_likelihood = likelihood.dot(self.weights) + epsilon
        return (likelihood * self.weights) / total_likelihood[:, np.newaxis]

    def _m_step(self, X, responsibilities):
        n_samples = X.shape[0]
        weights = responsibilities.sum(axis=0)
        self.weights = weights / n_samples
        epsilon = 1e-6  # A small constant for numerical stability
        # Update means with a small constant to prevent division by zero
        self.means = np.dot(responsibilities.T, X) / (weights[:, np.newaxis] + epsilon)
        for k in range(self.n_components):
            diff = X - self.means[k]
            # Update covariance matrices
            self.covariances[k] = np.dot((responsibilities[:, k, np.newaxis] * diff).T, diff) / (weights[k] + epsilon)

    def _pdf(self, X, component_idx):
        mean = self.means[component_idx]
        cov = self.covariances[component_idx]
        # Regularization to avoid singular covariance matrix
        reg_cov = cov + np.eye(cov.shape[0]) * 1e-6
        inv_cov = np.linalg.inv(reg_cov)
        det_cov = np.linalg.det(reg_cov)
        # Avoiding division by zero or negative determinant
        if det_cov < 1e-9:
            det_cov = 1e-9
        norm_const = np.sqrt(det_cov * (2 * np.pi) ** X.shape[1])
        return np.exp(-0.5 * np.sum((X - mean) @ inv_cov * (X - mean), axis=1)) / norm_const

    def _log_likelihood(self, X):
        pdf_values = np.array([self._pdf(X, k) for k in range(self.n_components)]).T
        # Ensure no zero values before dot product
        pdf_values = np.maximum(pdf_values, 1e-9)
        weighted_pdf = pdf_values.dot(self.weights)
        # Ensure no zero values before log
        weighted_pdf = np.maximum(weighted_pdf, 1e-9)
        return np.sum(np.log(weighted_pdf))

    def predict(self, X):
        # E-step to get responsibilities
        responsibilities = self._e_step(X)
        # Assign each sample to the cluster with the highest responsibility
        return np.argmax(responsibilities, axis=1)


if __name__ == "__main__":
    np.random.seed(5026)
    # mean1 = [2, 2]
    # cov1 = [[1, 0], [0, 1]]
    # data1 = np.random.multivariate_normal(mean1, cov1, 100)
    # mean2 = [8, 8]
    # cov2 = [[1.5, 0], [0, 1.5]]
    # data2 = np.random.multivariate_normal(mean2, cov2, 100)
    # data = np.vstack((data1, data2))
    # gmm = GaussianMixture(n_components=2)
    # gmm.fit(data)
    # print(gmm.predict(data))
    #
    X_train, y_train, X_test, y_test = get_dataset(train_num=None)
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_train[:, :3])
    print(gmm.predict(X_train[:, :3]).max())
    # print(gmm.covariances)
    # print(gmm.means)
