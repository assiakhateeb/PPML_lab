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
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X, Y)
# Visualizing the build Decision tree
from sklearn.tree import export_graphviz

""" gini is a measure that is used to find a node to split on it
    samples refer to the number of samples that are entering this node
    value is a split of the data instances according to their class 
     example: value = [instances_num_of_setosa, instances_num_of_versicolor, instances_num_of_virginica
    class is what the decision tree would predict
"""
# Creates dot file named tree.dot
export_graphviz(tree, out_file="myTree.dot", feature_names=list(X.columns),
                class_names=iris.target_names,
                filled=True, rounded=True)

# convert .dot file to png
""" https://onlineconvertfree.com/complete/dot-png/ """
