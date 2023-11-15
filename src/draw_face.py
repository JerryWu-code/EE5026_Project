from data_loader import image_to_mat
import numpy as np
import matplotlib.pyplot as plt
from config import image_format, train_dir, test_dir, output_fig_dir


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
        name = 'Eigen Face #{0}'.format(i + 1)
        ax.set_title(name)
        plt.show()
        if save_fig:
            fig.savefig(output_fig_dir + '{0}.png'.format(name))


if __name__ == "__main__":
    image_dir = train_dir

    train_image, new_labels, label_mapping = image_to_mat(image_dir=image_dir, target_num=500, use_selfie=True)
    draw_mean_face(train_image)
