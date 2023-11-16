import numpy as np
from config import image_format, train_dir, test_dir, output_fig_dir, seed
from data_loader import image_to_mat
import random
import matplotlib.pyplot as plt
from draw import draw_PCs_faces, draw_ProjectedData


def PCA(image_mat, num_PCs=3, method='normal'):
    mean_mat = np.mean(image_mat, axis=1).reshape(-1, 1)
    centered_mat = image_mat - mean_mat

    if method == 'normal':
        # normal: use SVD to replace the computation of S = X * X.T, eg: 1024*1024
        U, singular_values, Vt = np.linalg.svd(centered_mat, full_matrices=False)
        eigen_values = singular_values ** 2
        eigen_vectors = U[:, :num_PCs].dot(U[:, :num_PCs].T)
        # get all the eigen faces
        eigen_faces = eigen_vectors.dot(centered_mat)
        # get the information rate
        information_rate = sum(eigen_values[:num_PCs]) / sum(eigen_values)

    elif method == 'fast':
        # fast: use SVD to reduce the computational cost, so here we use L = X.T * X, eg: 500*500
        L = centered_mat.T.dot(centered_mat)
        eigen_values, eigen_vectors = np.linalg.eig(L)

        # sort the eigen_values and the corresponding eigen_vectors descending
        sorted_indices = np.argsort(eigen_values)[::-1]
        sorted_eigen_values = eigen_values[sorted_indices]
        sorted_eigen_vectors = eigen_vectors[:, sorted_indices]

        # get all the eigen faces
        eigen_faces = centered_mat.dot(sorted_eigen_vectors)
        # get the information rate
        information_rate = sum(sorted_eigen_values[:num_PCs]) / sum(sorted_eigen_values)

    print("After PCA with {0} PCs, the information rate is {1:.2f}%.".format(num_PCs, 100 * information_rate))

    # get the reduced eigen faces within num_PCs, eg: 1024 by 3
    reduced_eigen_faces = eigen_faces[:, :num_PCs] + mean_mat

    # project faces from original face-space to num_PCs-space
    proj_mat = reduced_eigen_faces.T.dot(centered_mat)  # 3 by 500

    return proj_mat, reduced_eigen_faces, information_rate


if __name__ == '__main__':
    # set the directory of the train image
    image_dir = train_dir

    # get the list of 500 samples matrix of training set images
    train_image, new_labels, label_mapping = image_to_mat(image_dir=image_dir, target_num=500, use_selfie=True,
                                                          seed=seed)
    # transform the train_image (500 by 32*32) to image_mat (1024 by 500)
    image_mat = np.array([np.ravel(i) for i in train_image]).T

    # implement PCA
    reduced_2d, reduced_eigen_faces_2d, _ = PCA(image_mat=image_mat, num_PCs=2, method='normal')
    reduced_3d, reduced_eigen_faces_3d, _ = PCA(image_mat=image_mat, num_PCs=3, method='normal')

    # draw and save the eigen faces
    # draw_PCs_faces(reduced_eigen_faces=reduced_eigen_faces_2d, save_fig=False)
    # draw_PCs_faces(reduced_eigen_faces=reduced_eigen_faces_3d, save_fig=False)

    draw_ProjectedData(reduced_2d, reduced_3d, new_labels=new_labels, selfie_label=25, save_fig=False, name='PCA')
