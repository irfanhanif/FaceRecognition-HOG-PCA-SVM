import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.set_printoptions(threshold='nan')

path = 'dataset/'
cls_list = []
feature_list = []
hog = cv2.HOGDescriptor()
size = (64, 128)

pca = PCA(n_components=50)

for file in os.listdir(path):
    if file.endswith(".jpg"):
        filename = str(file)

        cls = filename.split('.')[0]
        cls_list.append(cls)

        img = cv2.imread(path + filename)
        img = cv2.resize(img, size)
        feature = hog.compute(img)
        feature = [item[0] for item in feature]
        # pca.fit(feature)
        feature_list.append(feature)

pca.fit(feature_list)
# var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
# print var1.tolist().index(98.19000000000196)
# plt.plot(var1)
# plt.show()

y = np.array(cls_list)
X = np.array(pca.fit_transform(feature_list))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = svm.SVC().fit(X_train, y_train)
print clf.score(X_test, y_test) # return mean accuracy