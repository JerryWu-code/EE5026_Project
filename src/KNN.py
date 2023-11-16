import numpy as np
from data_loader import image_to_mat
from config import *
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def knn_classifier(X_train, y_train, X_test, k, y_test=None):
    """
    K-Nearest Neighbors (KNN) Classifier

    Parameters:
    - X_train (numpy.ndarray): Training data features (samples x features).
    - y_train (numpy.ndarray): Labels corresponding to training data (samples).
    - X_test (numpy.ndarray): New data points to classify (samples x features).
    - k (int): Number of nearest neighbors to consider for classification.
    - y_test (numpy.ndarray, optional): True labels for evaluating accuracy (samples).

    Returns:
    - predicted_classes (numpy.ndarray): Predicted class labels for the new data points (samples).
    - accuracy (float): Accuracy of the predictions (if y_true is provided).
    """

    # Calculate Euclidean distance function for two data points
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # Find the k nearest neighbors for a single data point
    def get_k_nearest_neighbors_single(X_train, y_train, x_new, k):
        distances = []
        for i in range(len(X_train)):
            distance = euclidean_distance(X_train[i], x_new)
            distances.append((X_train[i], y_train[i], distance))
        distances.sort(key=lambda x: x[2])
        return distances[:k]

    # Predict class based on majority vote
    def predict_class(neighbors):
        class_votes = {}
        for neighbor in neighbors:
            label = neighbor[1]
            if label in class_votes:
                class_votes[label] += 1
            else:
                class_votes[label] = 1
        sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
        return sorted_votes[0][0]

    # Initialize an empty array to store predicted classes
    predicted_classes = []

    # Iterate through each new data point in x_new
    for i in range(len(X_test)):
        new_sample = X_test[i]
        # Find k nearest neighbors for the current data point
        nearest_neighbors = get_k_nearest_neighbors_single(X_train, y_train, new_sample, k)
        # Predict class for the current data point and append it to the result array
        predicted_class = predict_class(nearest_neighbors)
        predicted_classes.append(predicted_class)

    # Convert the list of predicted classes to a numpy array
    predicted_classes = np.array(predicted_classes)

    # Calculate accuracy if y_true is provided
    accuracy = None
    if y_test is not None:
        correct_predictions = np.sum(predicted_classes == y_test)
        total_samples = len(y_test)
        accuracy = correct_predictions / total_samples

    return predicted_classes, accuracy


if __name__ == "__main__":
    image_dir1 = train_dir
    image_dir2 = test_dir

    # Train
    train_image, train_new_labels, train_label_mapping = image_to_mat(image_dir=image_dir1, target_num=500,
                                                                      use_selfie=True,
                                                                      seed=seed)
    train_image_mat = np.array([np.ravel(i) for i in train_image])  # 500 * 1024

    # Test
    test_image, test_new_labels, test_label_mapping = image_to_mat(image_dir=image_dir2, use_selfie=True, seed=seed)
    test_image_mat = np.array([np.ravel(i) for i in test_image])   # 1303 * 1024

    X_train = train_image_mat
    y_train = train_new_labels
    X_test = test_image_mat
    y_test = test_new_labels
    k = 3
    # predicted_classes, accuracy = knn_classifier(X_train, y_train, X_test, k, y_test)

    n_components = 20
    pca = PCA(n_components=n_components)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    predicted_classes, accuracy = knn_classifier(X_train_pca, y_train, X_test_pca, k, y_test)

    print("Predicted Classes:", predicted_classes)
    print("Accuracy:", accuracy)
