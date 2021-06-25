import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree


class Tree:

    def __init__(self, feature, threshold, leaf, lenHot, left, right):
        self.feature = feature
        self.threshold = threshold
        if leaf is not None:
            self.leaf = np.zeros(lenHot)
            self.leaf[leaf] = 1
        else:
            self.leaf = None
        self.left = left
        self.right = right

    def getSubTree(self, leftOrRight):
        if leftOrRight == 'Left':
            return self.left
        else:
            return self.right

    def getNode(self):
        return self.feature, self.threshold, self.leaf, self.left, self.right


def printTree(T, depth=0):
    if T is None:
        return
    feature, threshold, leaf, left, right = T.getNode()
    if isinstance(leaf, np.ndarray):
        print(depth * ' ', 'leaf = ', leaf)
    else:
        print(depth * ' ', 'feature = ', feature, 'threshold = ', threshold)
    printTree(left, depth + 4)
    printTree(right, depth + 4)
    return


def builtTree(clf, node_id=0):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left[node_id]
    # print('children_left=', clf.tree_.children_left[2])
    children_right = clf.tree_.children_right[node_id]
    feature = clf.tree_.feature[node_id]
    threshold = clf.tree_.threshold[node_id]
    lenHot = None
    is_split_node = children_left != children_right

    if is_split_node:
        leaf = None
        left = builtTree(clf, children_left)
        right = builtTree(clf, children_right)
        # print('split node id =', node_id)
    else:
        left = None
        right = None
        value = clf.tree_.value[node_id][0]
        # print(value)
        lenHot = len(value)
        leaf = np.argmax(value)
        # print(leaf)
        # print('leaf node id =', node_id)
    return Tree(feature, threshold, leaf, lenHot, left, right)


# iris = load_iris()
# X = iris.data
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
# clf.fit(X_train, y_train)
#
# DT = builtTree(clf)
# printTree(DT)
#
# fn = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# cn = ['setosa', 'versicolor', 'virginica']
#
# # Setting dpi = 300 to make image clearer than default
# # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=300)
# # fig = plt.figure(figsize=(25,20))
# fig, axes = plt.subplots(figsize=(4, 1), dpi=300)
# tree.plot_tree(clf,
#                feature_names=fn,
#                class_names=cn,
#                filled=True,
#                )
# plt.show()
