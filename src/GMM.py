import numpy as np
from data_loader import *
from PCA import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def GMMClusterings(X_train, y_train, title=None, n_comps: int = 3, show_samples: int = 10, iter_time=5, save_fig=True):
    """
    Apply GMM Clustering on a list of (preprocessed) training set images
    `None`:
    """
    for i in range(iter_time):
        # Fit the train data on a GMM and predict
        gmm = GaussianMixture(n_components=n_comps).fit(X_train)
        cls_pred = gmm.predict(X_train)
        print(cls_pred)
        # Randomly pick some images from each clusters to display
        cls_idxs_list = []
        for k in range(n_comps):
            cls_idxs = [j for j in range(cls_pred.shape[0]) if cls_pred[j] == k]
            cls_idxs_list.append(random.sample(cls_idxs, show_samples))

        # Plot some example faces in each cluster
        if show_samples > 0:
            n_rows = n_comps
            n_cols = show_samples
            img_shape = np.array([32, 32], dtype=int)
            fig = plt.figure(figsize=(16, 6))
            for i in range(n_rows):
                for j in range(n_cols):
                    ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1, xticks=[], yticks=[])
                    ax.imshow(X_train[cls_idxs_list[i][j]].reshape(img_shape), cmap='gray')
                    ax.set_xlabel('%d' % y_train[cls_idxs_list[i][j]])
                    if j == 0:
                        ax.set_ylabel('Cluster %d' % (i + 1))

            plt.show()
            if save_fig:
                fig.savefig(output_fig_dir + '{0}_GMM_3Component.png'.format(title, i))

    return


if __name__ == "__main__":
    np.random.seed(5026)
    # 1. Get data and pca
    X_train, y_train, X_test, y_test = get_dataset(train_num=None)
    train_reduced, reconstruct_faces, _, trans_mat = PCA(image_mat=X_train.T, num_PCs=200)  # 200
    X_train_pca = train_reduced.T
    X_test_pca = (np.dot(X_test - np.mean(X_test, axis=0), trans_mat))

    GMMClusterings(X_train, y_train, n_comps=3, show_samples=10, save_fig=True, title='200_PCA', iter_time=3)
