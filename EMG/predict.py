import os
import numpy as np
import math
from EMG.model.conv1D import Model
import torch
from collections import Counter


def get_data(txt_filepath):
    sku_data = []
    with open(txt_filepath, 'r', encoding='utf-8') as file:
        for line in file:
            number = int(line.strip())
            if number > 10000:
                continue
            sku_data.append(number / 10000)
    return sku_data


def get_image_feature(iemg):
    '''构建图像特征'''
    imageData = []
    imageLength = 200

    length = math.floor((iemg.shape[0] - imageLength) / imageLength)
    print(" number of sample: ", iemg.shape[0], length)

    for j in range(length):
        subImage = iemg[imageLength * j:imageLength * (j + 1)]
        imageData.append(subImage)

    return imageData


if __name__ == '__main__':
    # data process
    sku_data = get_data('./data/sku_data/5.txt')
    sku_data = np.array(sku_data, dtype=np.float32)
    image_data = get_image_feature(sku_data)

    model = Model(classes=6)
    model.load_state_dict(torch.load('./savemodel/epoch_83_0.8853503184713376_0.22168230170718745.pth'))

    results = []
    for image in image_data:
        data = np.expand_dims(image, axis=1)
        data = np.expand_dims(data, axis=2)
        data = np.expand_dims(data, axis=0)
        data = np.transpose(data, [0, -1, 1, 2])
        data = torch.from_numpy(data)
        data = data.type(torch.float32)
        prediction = model(data)
        results.append(torch.argmax(prediction, dim=-1)[0].item())

    counter = Counter(results)
    results = counter.most_common()
    print(results)
    print('{} 是最多的数，出现了 {}次'.format(results[0][0], results[0][1]))

    pass
