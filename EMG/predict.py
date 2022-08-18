import serial
import numpy as np
import math
from EMG.model.conv1D import Model
import torch
from collections import Counter
import time


def sample_data():
    # sets up serial connection (make sure baud rate is correct - matches Arduino)
    # 设置串口号和波特率和Arduino匹配
    ser = serial.Serial('com5', 115200)
    # a为储存数据的列表
    a = []

    while True:  # 30可以根据需要设置，while(True)：代表一直读下去
        # reads until it gets a carriage return. MAKE SURE THERE IS A CARRIAGE RETURN OR IT READS FOREVER

        data = ser.readline()  # 按行读取串口数据进来
        try:
            data = data.decode()  # 读进来的数据是bytes形式，需要转化为字符串格式
            data = data.strip()
            print(data)
            if len(data) == 0:
                break
            a.append(data)
        except Exception as e:
            print(e)
            break
    ser.close()
    with open('./test.txt', 'w', encoding='utf-8') as file:
        for number in a:
            file.write(number.strip() + '\n')


def get_data(txt_filepath):
    sku_data = []
    with open(txt_filepath, 'r', encoding='utf-8') as file:
        for line in file:
            number = int(line.strip())
            if number > 100:
                continue
            sku_data.append(number / 100)
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
    while True:
        print('开始采集数据')
        sample_data()
        time.sleep(0.5)

        # data process
        print('开始数据处理')
        sku_data = get_data('./test.txt')
        sku_data = np.array(sku_data, dtype=np.float32)
        image_data = get_image_feature(sku_data)

        print('load model')
        model = Model(classes=2)
        model.load_state_dict(torch.load('./savemodel/epoch_4_0.9032258064516129_0.3463459014892578.pth'))

        print('start predict')
        results = []
        for image in image_data:
            data = np.expand_dims(image, axis=1)
            data = np.expand_dims(data, axis=2)
            data = np.expand_dims(data, axis=0)
            data = np.transpose(data, [0, -1, 1, 2])
            data = torch.from_numpy(data)
            data = data.type(torch.float32)
            prediction = model(data)
            print(prediction)
            results.append(torch.argmax(prediction, dim=-1)[0].item())

        counter = Counter(results)
        results = counter.most_common()
        print(results)
        print('{} 是最多的数，出现了 {}次'.format(results[0][0], results[0][1]))
