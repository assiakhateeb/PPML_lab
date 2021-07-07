import numpy as np


class Node:
    def __init__(self, feature, threshold, index_of_max, lenHot, left, right, leaf):
        self.feature = feature
        self.threshold = threshold
        if leaf is not None:
            self.leaf = np.zeros(lenHot)
            self.leaf[index_of_max] = 1
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
    children_left = clf.tree_.children_left[node_id]
    children_right = clf.tree_.children_right[node_id]
    feature = clf.tree_.feature[node_id]
    threshold = clf.tree_.threshold[node_id]
    lenHot = None
    is_split_node = children_left != children_right
    index_of_max = -1
    if is_split_node:
        leaf = None
        left = builtTree(clf, children_left)
        right = builtTree(clf, children_right)
    else:
        leaf = True
        left = None
        right = None
        value = clf.tree_.value[node_id][0]
        lenHot = len(value)
        index_of_max = np.argmax(value)
    return Node(feature, threshold, index_of_max, lenHot, left, right, leaf)
