import os
from PIL import Image
import numpy as np
from config import *
import csv
import torch
import random
from collections import Counter
import pprint


def image_to_mat(image_dir, target_num=None, use_selfie=True, output_map=False, file=None, seed=seed):
    """
    Convert images to a matrix
    :param image_dir: Directory containing the images
    :param target_num: Number of target images (optional)
    :param use_selfie: Whether to include selfies (optional)
    :param output_map: Whether to output a label mapping file (optional)
    :param seed: Seed for randomization (optional)

    :return: Images: list of images, each image 32*32
             labels: list of labels for images
             label mapping: dict of old_to_new label mapping
    """
    # Initialize the list for images and labels
    images = []
    labels = []

    random.seed(seed)
    # reset the label-mappling in order, and the last one is selfie
    object_list = os.listdir(image_dir)
    if '.DS_Store' in object_list:
        object_list.remove('.DS_Store')

    if not target_num:
        target_num = sum(
            [1 for _, files in enumerate(os.walk(image_dir)) for i in files[2] if i.endswith(image_format)])

    # set whether we use selfie or not
    if not use_selfie:
        object_list.remove('selfie')
        average_class_num = int(target_num / len(object_list))
    else:
        selfie_num = sum([1 if i.endswith(image_format) else 0 for i in os.listdir(image_dir + '/selfie')])
        target_num = target_num - selfie_num
        average_class_num = round(target_num / (len(object_list) - 1))

    label_mapping = {name: index for index, name in enumerate(sorted(object_list))}

    # iterate every single folders corresponding to sub-objectives
    for folder_name in object_list:
        folder_path = os.path.join(image_dir, folder_name)

        if os.path.isdir(folder_path):
            # iterate every single image in the folder
            image_lst = [i for i in os.listdir(folder_path) if i.lower().endswith(image_format)]

            if folder_name == 'selfie':
                final_image_lst = image_lst
            elif folder_name == object_list[-1]:
                random.seed(seed)
                final_image_lst = random.sample(image_lst, target_num - average_class_num * (len(object_list) - 2))
            else:
                final_image_lst = random.sample(image_lst, average_class_num)

            for file_name in final_image_lst:
                img_path = os.path.join(folder_path, file_name)
                image = Image.open(img_path).convert('L').resize((32, 32))
                image_array = np.array(image)

                # add the image and label mapping to the list
                images.append(image_array)
                labels.append(label_mapping[folder_name])

    if output_map:
        new_dict = {}
        with open(file, 'w') as f:
            for key, value in label_mapping.items():
                new_value = {
                    'label': value,
                    'num': Counter(labels)[label_mapping[key]]
                }
                # file.write('{0}: {1}\n'.format(key, new_value))
                new_dict[key] = new_value
            f.write('Name: {0}\n\n'.format(file.split('/')[-1]))
            f.write(pprint.pformat(new_dict))
            f.write('\n\nTotal number: {0}\n'.format(len(labels)))

    return images, np.array(labels), label_mapping


def get_dataset(train_num=train_target_num):
    """
    Get the dataset and label mapping
    :param train_num: number of training images
    :return: for X, observations by features
             for y, observations vector
    """
    image_dir1 = train_dir
    image_dir2 = test_dir

    # Train
    train_image, train_new_labels, train_label_mapping = image_to_mat(image_dir=image_dir1, target_num=train_num,
                                                                      use_selfie=True,
                                                                      seed=seed)
    train_image_mat = np.array([np.ravel(i) for i in train_image])  # 500 * 1024
    # Test
    test_image, test_new_labels, test_label_mapping = image_to_mat(image_dir=image_dir2, use_selfie=True, seed=seed)
    test_image_mat = np.array([np.ravel(i) for i in test_image])  # 1303 * 1024

    X_train = train_image_mat
    y_train = train_new_labels
    X_test = test_image_mat
    y_test = test_new_labels

    return X_train, y_train, X_test, y_test


def save_loss_history_to_csv(loss_history, file_name):
    """
    Save the loss history to a CSV file.

    :param loss_history: List of loss values.
    :param file_name: Name of the file to save the loss history.
    """
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for loss in loss_history:
            writer.writerow([loss])


def read_loss_history_from_csv(file_name):
    """
    Read a loss history from a CSV file.

    :param file_name: Name of the file to read the loss history from.
    :return: List of loss values.
    """
    loss_history = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # Avoid empty rows
                loss_history.append(float(row[0]))
    return loss_history


def set_random_seed(seed_value):
    """
    Set the random seed for reproducible results.

    :param seed_value: An integer value for the seed.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


if __name__ == '__main__':
    image_dir1 = train_dir
    image_dir2 = test_dir
    save = False
    train_image, train_new_labels, train_label_mapping = image_to_mat(image_dir=image_dir1, target_num=train_target_num,
                                                                      use_selfie=True,
                                                                      output_map=save,
                                                                      file=PCA_train_dir, seed=seed)
    test_image, test_new_labels, test_label_mapping = image_to_mat(image_dir=image_dir2, use_selfie=True,
                                                                   output_map=save,
                                                                   file=PCA_test_dir, seed=seed)
    # print(train_image)
    # print(new_labels)
    get_dataset()
