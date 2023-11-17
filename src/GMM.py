import numpy as np

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

            # Check for convergence
            log_likelihood_new = self._log_likelihood(X)
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
        total_likelihood = likelihood.dot(self.weights)
        return (likelihood * self.weights) / total_likelihood[:, np.newaxis]

    def _m_step(self, X, responsibilities):
        n_samples = X.shape[0]
        weights = responsibilities.sum(axis=0)
        self.weights = weights / n_samples
        self.means = np.dot(responsibilities.T, X) / weights[:, np.newaxis]
        for k in range(self.n_components):
            diff = X - self.means[k]
            # Update covariance matrices
            self.covariances[k] = np.dot((responsibilities[:, k, np.newaxis] * diff).T, diff) / weights[k]

    def _pdf(self, X, component_idx):
        # Calculate the probability density function of a Gaussian
        mean = self.means[component_idx]
        cov = self.covariances[component_idx]
        return np.exp(np.dot(-0.5 * np.sum((X - mean), np.linalg.inv(cov) * (X - mean), axis=1))) / np.sqrt(
            np.linalg.det(cov) * (2 * np.pi) ** X.shape[1])

    def _log_likelihood(self, X):
        # Calculate log likelihood of the data under the current model
        return np.sum(np.log(np.array([self._pdf(X, k) for k in range(self.n_components)]).T.dot(self.weights)))

    def predict(self, X):
        # E-step to get responsibilities
        responsibilities = self._e_step(X)
        # Assign each sample to the cluster with the highest responsibility
        return np.argmax(responsibilities, axis=1)


if __name__ == "__main__":
    np.random.seed(0)

    mean1 = [2, 2]
    cov1 = [[1, 0], [0, 1]]
    data1 = np.random.multivariate_normal(mean1, cov1, 100)

    mean2 = [8, 8]
    cov2 = [[1.5, 0], [0, 1.5]]
    data2 = np.random.multivariate_normal(mean2, cov2, 100)

    data = np.vstack((data1, data2))

    gmm = GaussianMixture(n_components=2)
    gmm.fit(data)
    print(gmm.predict(data))
    print(gmm.covariances)
    print(gmm.means)