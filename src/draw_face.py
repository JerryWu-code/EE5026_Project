from data_loader import image_to_mat
import numpy as np
import matplotlib.pyplot as plt
from config import image_format, train_dir, test_dir, output_fig_dir

def draw_mean_face(mat):
    fig, ax = plt.subplots()

    mean_mat = np.mean(mat, axis=0)
    ax.imshow(mean_mat, cmap='gray')
    ax.axis('off')
    ax.set_title('Mean Face')

    plt.show()
    fig.savefig(output_fig_dir + 'Mean_Face.png')


if __name__ == "__main__":
    image_dir = train_dir

    train_image, new_labels, label_mapping = image_to_mat(image_dir=image_dir, target_num=500, use_selfie=True)
    draw_mean_face(train_image)