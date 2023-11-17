import numpy as np
from config import *
from data_loader import image_to_mat, get_dataset
from KNN import knn_classifier
from draw import *


class LDA:
    def __init__(self, k):
        self.lda_projection_matrix = None
        self.k = k

    def fit(self, X_train, y_train):
        """
        Fit the LDA model to the training data.
        Parameters:
        - X_train: numpy array, shape (n_train_samples, n_features)
          Training data.
        - y_train: numpy array, shape (n_train_samples,)
          Class labels for training data.
        """
        # Get unique class labels
        unique_labels = np.unique(y_train)
        num_classes = len(unique_labels)

        # Compute class means and within-class scatter matrix
        class_means = np.zeros((num_classes, X_train.shape[1]))
        Sw = np.zeros((X_train.shape[1], X_train.shape[1]))

        for i, label in enumerate(unique_labels):
            Xi = X_train[y_train == label]
            class_means[i] = np.mean(Xi, axis=0)
            Si = np.dot((Xi - class_means[i]).T, Xi - class_means[i])
            Sw += Si

        # Compute between-class scatter matrix
        Sb = np.zeros((X_train.shape[1], X_train.shape[1]))
        for i, label in enumerate(unique_labels):
            Ni = np.sum(y_train == label)
            mean_diff = class_means[i] - np.mean(X_train, axis=0)
            Sb += Ni * np.outer(mean_diff, mean_diff)

        # Calculate the SVD of the matrix Sw_inv dot Sb
        S = np.linalg.svd(np.linalg.inv(Sw).dot(Sb), full_matrices=False)

        # Choose the top k eigenvalues and corresponding eigenvectors
        top_eigenvalue_indices = np.argsort(S[1])[::-1][:self.k]
        top_eigenvectors = S[2][:, top_eigenvalue_indices]

        self.lda_projection_matrix = top_eigenvectors

    def transform(self, X_test, k=None):
        """
        Transform the test data using the LDA projection matrix.
        Parameters:
        - X_test: numpy array, shape (n_test_samples, n_features)
          Test data to be transformed.
        Returns:
        - X_test_lda: numpy array, shape (n_test_samples, k)
          Transformed data after dimensionality reduction using LDA.
        """
        if self.lda_projection_matrix is None:
            raise ValueError("LDA model has not been fit. Call 'fit' method first.")

        # Choose the top k eigenvectors
        top_k_eigenvectors = self.lda_projection_matrix[:, :self.k]

        # Project the test data onto the top k eigenvectors
        X_test_lda = np.dot(X_test, top_k_eigenvectors)

        return X_test_lda


if __name__ == "__main__":
    image_dir1 = train_dir
    image_dir2 = test_dir
    save = True

    X_train, y_train, X_test, y_test = get_dataset(train_target_num)

    # Use LDA reduce dimensionality to 2, 3, 9 respectively
    dimension_list = LDA_dimension_list
    selfie_indices = np.where(y_train == selfie_label)[0]
    cmupie_indices = np.where(y_train != selfie_label)[0]
    # example_indice = selfie_indices[case_selfie_num - 1]

    accuracy_list = []
    proj_2d_3d = []
    # example_face = [train_image_mat[:, example_indice].reshape(32, 32)]

    for i in range(2):
        group = [selfie_indices, cmupie_indices][i]
        for d in dimension_list:
            lda_model = LDA(k=d)
            lda_model.fit(X_train, y_train)
            X_train_lda = lda_model.transform(X_train)
            if d == 2:
                proj_2d_3d.append(X_train_lda.T)
            elif d == 3:
                proj_2d_3d.append(X_train_lda.T)
            X_test_lda = lda_model.transform(X_test)

            predicted_classes, accuracy = knn_classifier(X_train=X_train_lda[group, :], y_train=y_train[group],
                                                         X_test=X_test_lda[group, :], k=3, y_test=y_test[group])
            accuracy_list.append(accuracy)
            # if i == 0:
            #     example_face.append(reconstruct_faces[:, example_indice].reshape(32, 32))

    accu = np.array(accuracy_list).reshape(2, -1)
    # Draw projection space 2D & 3D
    draw_ProjectedData(proj_2d_3d[0], proj_2d_3d[1], new_labels=y_train,
                       selfie_label=selfie_label, save_fig=save, name='LDA')
    # Draw the accuracy curve
    draw_accuracy_curve(x=dimension_list, accu_mat=accu, save_fig=False, file_name='LDA & KNN')
