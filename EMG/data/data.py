import scipy.io as scio
import numpy as np
from EMG.data.feature_utils import featureRMS, featureMAV, featureSSC, featureWL, featureZC
import math
import h5py


def load_mat_file(mat_filepath):
    '''加载matlab文件'''
    data = scio.loadmat(mat_filepath)
    return data['emg'], data['label']


def get_valid_data_index(labels):
    '''加载有效的label对应的索引值'''
    index = []
    for i in range(len(labels)):
        if labels[i] != 0:
            index.append(i)
    return index


def get_time_feature(emg, label):
    '''将emg和label转化为目标的特征，便于进行分类'''
    featureData = []
    featureLabel = []
    classes = 16
    timeWindow = 200
    strideWindow = 200

    for i in range(classes):
        index = []
        for j in range(label.shape[0]):
            if (label[j, :] == i):
                index.append(j)
        iemg = emg[index, :]
        length = math.floor((iemg.shape[0] - timeWindow) / strideWindow)
        print("class ", i, ",number of sample: ", iemg.shape[0], length)

        for j in range(length):
            rms = featureRMS(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
            mav = featureMAV(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
            wl = featureWL(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
            zc = featureZC(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
            ssc = featureSSC(iemg[strideWindow * j:strideWindow * j + timeWindow, :])

            featureStack = np.hstack((rms, mav, wl, zc, ssc))

            featureData.append(featureStack)
            featureLabel.append(i)
    featureData = np.array(featureData)

    return featureData, featureLabel


def get_image_feature(emg, label):
    '''构建图像特征'''
    imageData = []
    imageLabel = []
    classes = 16
    imageLength = 200

    for i in range(classes):
        index = []
        for j in range(label.shape[0]):
            if (label[j, :] == i):
                index.append(j)

        iemg = emg[index, :]
        length = math.floor((iemg.shape[0] - imageLength) / imageLength)
        print("class ", i, " number of sample: ", iemg.shape[0], length)

        for j in range(length):
            subImage = iemg[imageLength * j:imageLength * (j + 1), :]
            imageData.append(subImage)
            imageLabel.append(i)

    imageData = np.array(imageData)
    return imageData, imageLabel


def save_time_feature(save_filepath, time_featureData, time_featureLabel):
    file = h5py.File(save_filepath, 'w')
    file.create_dataset('featureData', data=time_featureData)
    file.create_dataset('featureLabel', data=time_featureLabel)
    file.close()


def save_image_feature(save_filepath, image_featureData, image_featureLabel):
    file = h5py.File(save_filepath, 'w')
    file.create_dataset('featureData', data=image_featureData)
    file.create_dataset('featureLabel', data=image_featureLabel)
    file.close()


if __name__ == '__main__':
    e_1_emg, e_1_label = load_mat_file('origen_dataset/S1_E1.mat')

    e_1_index = get_valid_data_index(e_1_label)
    e_1_emg, e_1_label = e_1_emg[e_1_index, :], e_1_label[e_1_index, :]

    e_2_emg, e_2_label = load_mat_file('origen_dataset/S1_E2.mat')
    e_2_index = get_valid_data_index(e_2_label)
    e_2_emg, e_2_label = e_2_emg[e_2_index, :], e_2_label[e_2_index, :]
    e_2_label = e_2_label + e_1_label[-1, :]

    e_3_emg, e_3_label = load_mat_file('origen_dataset/S1_E3.mat')
    e_3_index = get_valid_data_index(e_3_label)
    e_3_emg, e_3_label = e_3_emg[e_3_index, :], e_3_label[e_3_index, :]
    e_3_label = e_3_label + e_2_label[-1, :]

    e_4_emg, e_4_label = load_mat_file('origen_dataset/S1_E4.mat')
    e_4_index = get_valid_data_index(e_4_label)
    e_4_emg, e_4_label = e_4_emg[e_4_index, :], e_4_label[e_4_index, :]
    e_4_label = e_4_label + e_3_label[-1, :]

    # fuse sku_data,concat
    emg = np.vstack((e_1_emg, e_2_emg, e_3_emg, e_4_emg))
    label = np.vstack((e_1_label, e_2_label, e_3_label, e_4_label))
    label = label - 1

    time_featureData, time_featureLabel = get_time_feature(emg, label)
    image_featureData, image_featureLabel = get_image_feature(emg, label)

    save_time_feature('time_feature.h5', time_featureData, time_featureLabel)
    save_image_feature('image_feature.h5', image_featureData, image_featureLabel)
    # pass
