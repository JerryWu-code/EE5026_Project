import os
import random

def split_dataset(base_dir, output_dir, train_ratio=0.7, num_object=50):
    folders = os.listdir(base_dir)
    selected_folders = random.sample(folders, num_object)

    # set the train & test images folder path
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for folder in selected_folders:
        if folder != ".DS_Store":
            folder_path = os.path.join(base_dir, folder)
        images = os.listdir(folder_path)

        # randomly reorder the images
        random.shuffle(images)

        # split the train & test set according to the ratio
        split_index = int(len(images) * train_ratio)

        # assign images to train or test folder
        for i, image in enumerate(images):
            src_path = os.path.join(folder_path, image)

            # set the file name
            new_image_name = "{}".format(image)
            os.makedirs(os.path.join(train_dir, folder), exist_ok=True)
            os.makedirs(os.path.join(test_dir, folder), exist_ok=True)

            if i < split_index:
                dst_path = os.path.join(train_dir, folder, new_image_name)
            else:
                dst_path = os.path.join(test_dir, folder, new_image_name)

            # output the image
            with open(src_path, 'rb') as src_file:
                with open(dst_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())

if __name__ == "__main__":
    base_dir = '../data/PIE/'
    output_dir = '../data/'
    split_dataset(base_dir=base_dir, output_dir=output_dir, train_ratio=0.7, num_object=25)
