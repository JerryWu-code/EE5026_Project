import numpy as np
from config import *
from data_loader import image_to_mat
import random
import matplotlib.pyplot as plt
from draw import *
from KNN import knn_classifier


def PCA(image_mat, num_PCs=3, write_result=False, file_name=PCA_train_dir):
    mean_mat = np.mean(image_mat, axis=1).reshape(-1, 1)
    centered_mat = image_mat - mean_mat

    # use SVD to replace the computation of S = X * X.T, eg: 1024*1024
    U, singular_values, Vt = np.linalg.svd(centered_mat, full_matrices=False)
    eigen_values = singular_values ** 2
    eigen_vectors = U[:, :num_PCs]
    reconstruct_faces = np.dot(eigen_vectors.dot(eigen_vectors.T), centered_mat) + mean_mat

    proj_mat = np.dot(eigen_vectors.T, centered_mat)
    # get the information rate
    information_rate = sum(eigen_values[:num_PCs]) / sum(eigen_values)

    print_result = "After PCA with {0} PCs, the information rate is {1:.2f}%.\n".format(num_PCs, 100 * information_rate)
    if write_result:
        with open('../data/' + file_name, 'a') as file:
            file.write(print_result)
    print(print_result)

    return proj_mat, reconstruct_faces, information_rate, eigen_vectors


if __name__ == '__main__':
    # set the directory of the train & test image
    image_dir1 = train_dir
    image_dir2 = test_dir
    save = False

    # 1. Get data
    #   1.1 train set
    #       get the list of 500 samples matrix of training set images
    train_image, train_new_labels, _ = image_to_mat(image_dir=image_dir1, target_num=500,
                                                                      use_selfie=True,
                                                                      seed=seed)
    #       transform the train_image (500 by 32*32) to image_mat (1024 by 500)
    train_image_mat = np.array([np.ravel(i) for i in train_image]).T

    #   1.2 test set
    #       get the list of all 1303 samples matrix of testing set images
    test_image, test_new_labels, _ = image_to_mat(image_dir=image_dir2, use_selfie=True, seed=seed)
    #       transform the test_image (1303 by 32*32) to image_mat (1024 by 1303)
    test_image_mat = np.array([np.ravel(i) for i in test_image]).T

    # 2. Implement PCA
    train_reduced_2d, _, _, train_reduced_eigen_faces_2d = PCA(image_mat=train_image_mat, num_PCs=2,
                                                               write_result=save, file_name=PCA_train_dir)
    train_reduced_3d, _, _, train_reduced_eigen_faces_3d = PCA(image_mat=train_image_mat, num_PCs=3,
                                                               write_result=save, file_name=PCA_train_dir)

    # 3. Draw and save the eigen faces and projection results of training set.
    draw_PCs_faces(reduced_eigen_faces=train_reduced_eigen_faces_3d, save_fig=save)
    draw_ProjectedData(train_reduced_2d, train_reduced_3d, new_labels=train_new_labels,
                       selfie_label=selfie_label, save_fig=save, name='PCA')

    # 4. Use PCA reduce dimensionality to 40, 80, 200 respectively
    k = 3
    dimension_list = PCA_dimension_list
    selfie_indices = np.where(train_new_labels == selfie_label)[0]
    cmupie_indices = np.where(train_new_labels != selfie_label)[0]
    example_indice = selfie_indices[case_selfie_num - 1]

    accuracy_list = []
    example_face = [train_image_mat[:, example_indice].reshape(32, 32)]

    for i in range(2):
        group = [selfie_indices, cmupie_indices][i]
        y_train = train_new_labels[group]
        y_test = test_new_labels[group]
        for d in dimension_list:
            train_reduced, reconstruct_faces, _, trans_mat = PCA(image_mat=train_image_mat, num_PCs=d,
                                                                 write_result=save, file_name=PCA_train_dir)
            X_train_pca = train_reduced.T[group]
            X_test_pca = (np.dot(test_image_mat.T - np.mean(test_image_mat.T, axis=0), trans_mat))[group]

            predicted_classes, accuracy = knn_classifier(X_train_pca, y_train, X_test_pca, k, y_test)
            accuracy_list.append(accuracy)
            if i == 0:
                example_face.append(reconstruct_faces[:, example_indice].reshape(32, 32))

    accu = np.array(accuracy_list).reshape(2, -1)
    # show reconstructed face of my selfie

    # 5. Draw the Reconstruction Faces and Accuracy Curve
    draw_faces(faces_mat=np.array(example_face), sub_width=4, save_fig=save,
               title_lst=['Selfie', 'D={}'.format(dimension_list[0]), 'D={}'.format(dimension_list[0]),
                          'D={}'.format(dimension_list[0])],
               file_name='PCA_Reconstru_selfie_{}'.format(case_selfie_num))
    draw_accuracy_curve(x=dimension_list, accu_mat=accu, save_fig=True, file_name='PCA & KNN')
