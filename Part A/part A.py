# jupyter-lab
# from sklearn import tree
# X = [[0, 0], [1, 1]]
# Y = [0, 1]
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, Y)
# tree.plot_tree(clf)

# from sklearn.datasets import load_iris
# from sklearn import tree
#
# iris = load_iris()
# X, y = iris.data, iris.target
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, y)
# tree.plot_tree(clf)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
print(iris)
X = pd.DataFrame(iris.data[:, :], columns=iris.feature_names[:])
print(X)
Y = pd.DataFrame(iris.target, columns=["Species"])
print(Y)
# Now let's fit a DecisionTreeClassifier instance
tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X,Y)