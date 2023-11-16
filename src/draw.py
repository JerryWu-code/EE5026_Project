from data_loader import image_to_mat
import numpy as np
import matplotlib.pyplot as plt
import math
from config import image_format, train_dir, test_dir, output_fig_dir


def draw_faces(faces_mat, sub_width=None, save_fig=False, name=None):
    """
    Draw the faces of the dataset.
    :param name: filename of the output figure.
    :param sub_width: fig subplot width, number of cols.
    :param save_fig: whether save the face file.
    :param faces_mat: 2D array of images, number * 32 * 32.
    :return: figure
    """
    length = math.ceil(faces_mat.shape[0] / sub_width)
    fig, ax = plt.subplots(length, sub_width)  # , figsize=(10, 10))

    for i in range(length * sub_width):
        if length == 1:
            axi = ax[i]
        else:
            row = i // sub_width
            col = i - row * sub_width
            axi = ax[row, col]

        if i + 1 <= faces_mat.shape[0]:
            axi.imshow(faces_mat[i], cmap='gray')
            axi.axis('off')
            axi.set_title(i+1)
        else:
            axi.set_visible(False)

    plt.show()
    if save_fig:
        fig.savefig(output_fig_dir + '{0}Faces.png'.format(name + ': '))
    return fig


def draw_mean_face(mat, save_fig=False):
    """
    Draw the mean face of the dataset.
    :param save_fig: whether save the mean face file.
    :param mat: list of 2 Dimensional array of images.
    :return: figure
    """
    fig, ax = plt.subplots()

    mean_mat = np.mean(mat, axis=0)
    ax.imshow(mean_mat, cmap='gray')
    ax.axis('off')
    ax.set_title('Mean Face')

    plt.show()
    if save_fig:
        fig.savefig(output_fig_dir + 'Mean_Face.png')
    return fig


def draw_PCs_faces(reduced_eigen_faces, save_fig=False):
    """
    Draw the first num_PCs principal components of the reduced eigen faces.
    :param save_fig: whether save the eigen faces files.
    :param reduced_eigen_faces: list of 2 Dimensional array of images, origin_pixels by num_PCs.
    :return: figure
    """
    num_PCs = reduced_eigen_faces.shape[1]

    for i in range(num_PCs):
        fig, ax = plt.subplots()

        image_data = reduced_eigen_faces[:, i].reshape(32, 32)

        ax.imshow(image_data, cmap='gray')
        ax.axis('off')
        name = '{1}PCs: Eigen Face #{0}'.format(i + 1, num_PCs)
        ax.set_title(name)
        plt.show()
        if save_fig:
            fig.savefig(output_fig_dir + '{0}.png'.format(name))


if __name__ == "__main__":
    image_dir = train_dir

    train_image, new_labels, label_mapping = image_to_mat(image_dir=image_dir, target_num=500, use_selfie=True)
    # draw_mean_face(train_image)

    selfie_indices = np.where(np.array(new_labels) == 25)[0]
    draw_faces(faces_mat=np.array(train_image)[selfie_indices], sub_width=4, save_fig=False, name=None)
