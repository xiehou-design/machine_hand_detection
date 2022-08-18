import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

file = h5py.File('../data/time_feature.h5', 'r')
featureData = file['featureData'][:]
featureLabel = file['featureLabel'][:]
file.close()

featureData = MinMaxScaler().fit_transform(featureData)  # 缩放到[0, 1]
train_x, test_x, train_y, test_y = train_test_split(featureData, featureLabel, test_size=0.1)

SVM = SVC(C=2, kernel='rbf', degree=3, gamma=2)
SVM.fit(train_x, train_y)

score = SVM.score(train_x, train_y)
predict_SVM = SVM.predict(test_x)
accuracy_SVM = metrics.accuracy_score(test_y, predict_SVM)
print(score, accuracy_SVM)
