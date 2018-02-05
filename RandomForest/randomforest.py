from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

clf = RandomForestClassifier(n_estimators=10)
iris = load_iris()
clf.fit(iris.data, iris.target)

predict=clf.predict([[0.1,0.2,0.3,0.4]])
print(predict)
