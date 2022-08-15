import os
import random

image_names = os.listdir('./hand1')

random.shuffle(image_names)
train_image_names = image_names[len(image_names) // 10:]
test_image_names = image_names[:len(image_names) // 10]

with open('./train.txt', 'w', encoding='utf-8') as file:
    for image_name in train_image_names:
        file.write(image_name + '\n')

with open('./test.txt', 'w', encoding='utf-8') as file:
    for image_name in test_image_names:
        file.write(image_name + '\n')
