import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

file = h5py.File('../sku_data/time_feature.h5', 'r')
featureData = file['featureData'][:]
featureLabel = file['featureLabel'][:]
file.close()

featureData = MinMaxScaler().fit_transform(featureData)  # 缩放到[0, 1]
train_x, test_x, train_y, test_y = train_test_split(featureData, featureLabel, test_size=0.1)

RF = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=None, min_samples_split=2,
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                            max_leaf_nodes=None, min_impurity_decrease=0.0,
                            bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=0,
                            warm_start=False, class_weight=None)

RF.fit(train_x, train_y)
score = RF.score(train_x, train_y)
predict = RF.predict(test_x)
accuracy = metrics.accuracy_score(test_y, predict)

print("RF train accuracy: %.2f%%" % (100 * score))
print('RF test  accuracy: %.2f%%' % (100 * accuracy))
