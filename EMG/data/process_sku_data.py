import os
import numpy as np
import math
import h5py

sku_data = []
sku_label = []


def get_data(txt_filepath, label):
    with open(txt_filepath, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                number = int(line.strip())
                if number > 100:
                    continue
                sku_data.append(number)
                sku_label.append(label)
            except:
                pass


def get_image_feature(emg, label):
    '''构建图像特征'''
    imageData = []
    imageLabel = []
    classes = 2
    imageLength = 200

    for i in range(classes):
        index = []
        for j in range(label.shape[0]):
            if (label[j] == i):
                index.append(j)

        iemg = emg[index]
        length = math.floor((iemg.shape[0] - imageLength) / imageLength)
        print("class ", i, " number of sample: ", iemg.shape[0], length)

        for j in range(length):
            subImage = iemg[imageLength * j:imageLength * (j + 1)]
            imageData.append(subImage)
            imageLabel.append(i)

    imageData = np.array(imageData)
    return imageData, imageLabel


def save_image_feature(save_filepath, image_featureData, image_featureLabel):
    file = h5py.File(save_filepath, 'w')
    file.create_dataset('featureData', data=image_featureData)
    file.create_dataset('featureLabel', data=image_featureLabel)
    file.close()


if __name__ == '__main__':
    # get_data('./sku_data/ok.txt', 0)
    # get_data('./sku_data/8.txt', 1)
    # get_data('./sku_data/2.txt', 2)
    # get_data('./sku_data/3.txt', 3)
    # # get_data('./sku_data/4.txt', 4)
    # get_data('./sku_data/5.txt', 4)
    get_data('./sku_data/quan.txt', 0)
    get_data('./sku_data/bu.txt', 1)
    sku_data = np.array(sku_data, dtype=np.int32)
    sku_label = np.array(sku_label, np.int32)

    image_data, label_data = get_image_feature(sku_data, sku_label)
    print(image_data.shape)
    save_image_feature('./sku_data/image_feature.h5', image_data, label_data)
