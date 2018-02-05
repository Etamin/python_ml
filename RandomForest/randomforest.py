from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score

#random forest
clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
X, Y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)
scores=cross_val_score(clf,X,Y)
print scores.mean()

#extra tree
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
X, Y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)
scores=cross_val_score(clf,X,Y)
print scores.mean()
