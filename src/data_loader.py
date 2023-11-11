import os
from PIL import Image
import numpy as np
from config import image_format, train_dir, test_dir
import random
from collections import Counter
import pprint

def image_to_mat(image_dir, target_num=None, use_selfie=True, output_map=False, map_name=None):
    # Initialize the list for images and labels
    images = []
    labels = []

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
        with open('../data/' + map_name, 'w') as file:
            for key, value in label_mapping.items():
                new_value = {
                    'label': value,
                    'num': Counter(labels)[label_mapping[key]]
                }
                # file.write('{0}: {1}\n'.format(key, new_value))
                new_dict[key] = new_value
            file.write('Name: {0}\n\n'.format(map_name))
            file.write(pprint.pformat(new_dict))
            file.write('\n\nTotal number: {0}'.format(len(labels)))

    return images, labels, label_mapping


if __name__ == '__main__':
    image_dir = train_dir
    train_image, new_labels, label_mapping = image_to_mat(image_dir=image_dir, target_num=500, use_selfie=True, output_map=True,
                                                map_name='PCA_train_map.txt')
    # print(train_image)
    # print(new_labels)
    print(train_image[0], train_image[0].shape, type(train_image[0]))