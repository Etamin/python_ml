#coding:utf-8

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

iris = datasets.load_iris()
iris_data = iris.data
iris_target = iris.target
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.2)
model = SVC().fit(x_train, y_train)
predict = model.predict(x_test)
right = sum(predict == y_test)
print ('acc：%f%%' % (right * 100.0 / predict.shape[0]))
