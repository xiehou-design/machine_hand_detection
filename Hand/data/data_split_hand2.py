import os
import random

image_dirs = os.listdir('./hand2')

image_filepaths = []
for image_dir in image_dirs:
    new_dir_path = os.path.join('./hand2', image_dir)
    images = os.listdir(new_dir_path)
    _images = [image_dir + '/' + image for image in images]
    image_filepaths += _images

random.shuffle(image_filepaths)
train_image_names = image_filepaths[len(image_filepaths) // 10:]
test_image_names = image_filepaths[:len(image_filepaths) // 10]

with open('./train.txt', 'w', encoding='utf-8') as file:
    for image_name in train_image_names:
        file.write(image_name + '\n')

with open('./test.txt', 'w', encoding='utf-8') as file:
    for image_name in test_image_names:
        file.write(image_name + '\n')
