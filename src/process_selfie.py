import os
from PIL import Image
import random
from config import train_dir, test_dir, raw_selfie_dir, final_selfie_dir

def resize_and_convert_to_grayscale(input_path, output_path, split=False, test_ratio=0.7):
    images = os.listdir(input_path)
    if ".DS_Store" in images:
        images.remove(".DS_Store")
    # open image
    for i, image in enumerate(images):
        input_image_path = os.path.join(input_path, image)
        output_image_path = os.path.join(output_path, image)
        with Image.open(input_image_path) as img:
            # turn into grey scale
            gray_img = img.convert('L')

            # 32x32 resolution
            resized_img = gray_img.resize((32, 32))

            # save it
            resized_img.save(output_image_path)

    if split:
        grey_images = os.listdir(output_path)
        if ".DS_Store" in grey_images:
            grey_images.remove(".DS_Store")
        random.shuffle(grey_images)

        os.makedirs(os.path.join(train_dir, 'selfie/'), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'selfie/'), exist_ok=True)

        for i, grey_image in enumerate(grey_images):
            input_grey_image_path = os.path.join(output_path, grey_image)
            if i < int(len(grey_images) * test_ratio):
                output_grey_image_path = os.path.join(os.path.join(train_dir, 'selfie/'), grey_image)
            else:
                output_grey_image_path = os.path.join(os.path.join(test_dir, 'selfie/'), grey_image)
            with Image.open(input_grey_image_path) as img:
                img.save(output_grey_image_path)

if __name__ == "__main__":
    # transform images into 32*32 grey scale and split them into train & test set
    input_path = raw_selfie_dir
    output_path = final_selfie_dir
    os.makedirs(output_path, exist_ok=True)
    resize_and_convert_to_grayscale(input_path, output_path, split=True)
