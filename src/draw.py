from data_loader import *
import numpy as np
import matplotlib.pyplot as plt
import math
from config import *


def draw_faces(faces_mat, sub_width=None, save_fig=False, title_lst=None, file_name=None):
    """
    Draw the faces of the dataset.
    :param title_lst: title list of the subplots.
    :param file_name: filename of the output figure.
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
            if not title_lst:
                axi.set_title(i + 1)
            else:
                axi.set_title(title_lst[i])
        else:
            axi.set_visible(False)

    plt.show()
    if save_fig:
        fig.savefig(output_fig_dir + '{0}Faces.png'.format(file_name + ': '))
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
        name = 'Eigen Face #{0}'.format(i + 1)
        ax.set_title(name)
        plt.show()
        if save_fig:
            fig.savefig(output_fig_dir + '{0}.png'.format(name))


def draw_ProjectedData(reduced_2d, reduced_3d, new_labels, selfie_label=25, save_fig=False, name=None):
    """
    Plot the projected data onto 2D and 3D scatter plots respectively.
    :param reduced_2d: Array containing the data projected onto 2 dimensions.
                       Each column represents a data point.
    :param reduced_3d: Array containing the data projected onto 3 dimensions.
                       Each column represents a data point.
    :param new_labels: Array of labels for each data point. Used to differentiate
                       different categories in the plot.
    :param selfie_label: The label assigned to the 'selfie' category. Default is 25.
                         Data points with this label will be plotted with a different color.
    :param save_fig: Boolean flag indicating whether to save the plot to a file.
                     If True, the plot will be saved.
    :param name: Optional string representing the name of the projection method used
                 (e.g., 'PCA', 'LDA'). This is used for the plot title and for saving the file.

    If 'save_fig' is True, the plot will be saved to the path specified by
    'output_fig_dir' with the file name 'Projection of {name}.png'.
    """
    selfie_indices = np.where(new_labels == selfie_label)[0]
    cmupie_indices = np.where(new_labels != selfie_label)[0]
    proj_PIE_2d = np.array(reduced_2d)[:, cmupie_indices]
    proj_ME_2d = np.array(reduced_2d)[:, selfie_indices]
    proj_PIE_3d = np.array(reduced_3d)[:, cmupie_indices]
    proj_ME_3d = np.array(reduced_3d)[:, selfie_indices]

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # 2D Plot
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(proj_PIE_2d[0, :], proj_PIE_2d[1, :], s=10, c='cornflowerblue', label='CMU PIE')
    ax.scatter(proj_ME_2d[0, :], proj_ME_2d[1, :], s=15, c='r', label='Selfie')
    ax.set_xlabel('Principle Component 1')
    ax.set_ylabel('Principle Component 2')
    ax.set_title('2D Projection')
    ax.legend()
    # 3D Plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(proj_PIE_3d[0, :], proj_PIE_3d[1, :], proj_PIE_3d[2, :], s=10, c='cornflowerblue', label='CMU PIE')
    ax.scatter(proj_ME_3d[0, :], proj_ME_3d[1, :], proj_ME_3d[2, :], s=15, c='r', label='Selfie')
    ax.set_xlabel('Principle Component 1')
    ax.set_ylabel('Principle Component 2')
    ax.set_zlabel('Principle Component 3')
    ax.set_title('3D Projection')
    if name == 'LDA':
        ax.view_init(azim=120, elev=20)  # LDA
    elif name == 'PCA':
        ax.view_init(azim=150, elev=20)  # PCA
    ax.legend()
    plt.show()
    if save_fig:
        fig.savefig(output_fig_dir + 'Projection of {0}.png'.format(name))


def draw_accuracy_curve(x, accu_mat: np.array, save_fig=False, file_name=None):
    """
    Plot the accuraty graph based on the given data.
    :param:
    accu_mat: row 1~Selfie, row 2~CMU PIE
    save_fig: whether save this
    """
    fig, ax = plt.subplots()
    ax.plot(x, 100 * accu_mat[1, :], marker='o', color='cornflowerblue', label='CMU PIE', linestyle='-')
    ax.plot(x, 100 * accu_mat[0, :], marker='*', color='r', label='Selfie', linestyle=':')
    ax.set_xlabel('Image Dimensions')
    ax.set_ylabel('Classification Accuracy (%)')
    ax.legend(loc='best')
    ax.set_title('{} Accuracy Curve in Testing Set'.format(file_name))
    for i, (xi, yi1, yi2) in enumerate(zip(x, 100 * accu_mat[1, :], 100 * accu_mat[0, :])):
        ax.text(xi, yi1 - 5, '{:.2f}%'.format(yi1), ha='center', va='bottom', color='cornflowerblue')  # -3.5
        ax.text(xi, yi2 + 2, '{:.2f}%'.format(yi2), ha='center', va='top', color='r')  # +2
    plt.show()
    if save_fig:
        fig.savefig(output_fig_dir + '{0}: Accuracy_Curve.png'.format(file_name))


def plot_loss_history(loss_histories, labels, title='Training Loss History',
                      xlabel='Iterations', ylabel='Loss', save_fig=False):
    """
    Plot the training loss history for different models.

    :param save_fig: whether save the figure or not.
    :param loss_histories: A list of loss histories (list of lists).
    :param labels: A list of labels for each loss history.
    :param title: Title of the plot.
    :param xlabel: Label for the X-axis.
    :param ylabel: Label for the Y-axis.
    """
    fig = plt.figure(figsize=(12, 6))
    for loss_history, label in zip(loss_histories, labels):
        plt.plot(loss_history, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    if save_fig:
        fig.savefig(output_fig_dir + '{0}.png'.format(title))


if __name__ == "__main__":
    image_dir = train_dir
    save = True

    train_image, new_labels, label_mapping = image_to_mat(image_dir=image_dir, target_num=train_target_num,
                                                          use_selfie=True)
    # 1. Draw the mean face.
    # draw_mean_face(train_image, save_fig=save)

    # 2. Draw the selfie faces.
    selfie_indices = np.where(np.array(new_labels) == 25)[0]
    draw_faces(faces_mat=np.array(train_image)[selfie_indices], sub_width=4, save_fig=save,
               file_name='My_Selfie')

    # 3. Draw the training loss of CNN and ResNet18.
    loss_history_model1 = read_loss_history_from_csv(cnn_loss_dir)
    loss_history_model2 = read_loss_history_from_csv(resnet18_loss_dir)
    plot_loss_history(loss_histories=[loss_history_model2, loss_history_model1],
                      labels=['ResNet-18', 'CNN'], save_fig=save)
