# jupyter-lab
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
from my_tree import *

iris = load_iris()
X = pd.DataFrame(iris.data[:, :], columns=iris.feature_names[:])

""" preprocessing 1 - Feature scaling """
# rescale a range between an arbitrary set of values [a, b] where a=-1, b=1
scaler = MinMaxScaler(feature_range=(-1, 1))
X_rescaled_features = scaler.fit_transform(X)
X_rescaled_features = pd.DataFrame(X_rescaled_features[:, :], columns=iris.feature_names[:])
# print(X_rescaled_features)

Y = pd.DataFrame(iris.target, columns=["Species"])
# print(Y)
""" preprocessing 1 - Feature scaling """
Y_rescaled_features = scaler.fit_transform(Y)
Y_rescaled_features = pd.DataFrame(Y_rescaled_features, columns=["Species"])
# print(Y_rescaled_features)

X_train, X_test, Y_train, Y_test = train_test_split(X_rescaled_features, Y_rescaled_features, test_size=0.25)
# Now let's fit a DecisionTreeClassifier instance
Dtree = DecisionTreeClassifier(max_depth=2)
Dtree.fit(X_train, Y_train)

"""calculate the score"""
s = Dtree.score(X_test, Y_test)
# print("\n", "score=", s, "\n")


' Making a Prediction on a new sample '
sample_data1 = int(Dtree.predict([[5, 5, 1, 3]]))
# print(iris.target_names[sample_data1])

sample_data1 = int(Dtree.predict([[5, 5, 2.6, 1.5]]))
# print(iris.target_names[sample_data1])

sample_data1 = int(Dtree.predict([[5, 5, 15, 5]]))
# print(iris.target_names[sample_data1])

# tree.plot_tree(Dtree)
# plt.show()

# Visualizing the build Decision tree with dot file
#     gini is a measure that is used to find a node to split on it
#     samples refer to the number of samples that are entering this node
#     value is a split of the data instances according to their class
#     value = [instances_num_of_setosa, instances_num_of_versicolor, instances_num_of_virginica]
#     class is what the decision tree would predict

# fn = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# cn = ['setosa', 'versicolor', 'virginica']
#
# # Setting dpi = 300 to make image clearer than default
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
#
# tree.plot_tree(Dtree,
#                feature_names=fn,
#                class_names=cn,
#                filled=True);
# plt.show()


values = np.array(Dtree.tree_.value)
ohe = []
for val in values:
    # print("value= ", val)
    one_hot_encoding = np.zeros_like(val)
    one_hot_encoding[np.arange(len(val)), val.argmax(1)] = 1
    ohe.append(one_hot_encoding)
    # print("enc= ", one_hot_encoding)


def find_leaves(X, clf):
    """A function to find leaves of a DecisionTreeClassifier
    clf must be a fitted DecisionTreeClassifier
    """
    return set(clf.apply(X))


leaves = find_leaves(X_train, Dtree)
# print(leaves)
leaves_value_list = []
# for val in values:
for l in leaves:
    leaves_value_list.append(Dtree.tree_.value[l])
leaves_value_list = np.array(leaves_value_list)
# print(leaves_value_list)

ohe = []
for val in leaves_value_list:
    one_hot_encoding = np.zeros_like(val)
    one_hot_encoding[np.arange(len(val)), val.argmax(1)] = 1
    ohe.append(one_hot_encoding)

ohe = np.array(ohe)


# print(ohe)

class T:
    tree = clone(Dtree)
    ohe_leaf_value = ohe


mytree = T()
mytree.tree.fit(X_train, Y_train)

features = mytree.tree.tree_.feature
# print("feature=", features)
# features = set(features)
# print( "f=",features)
# print(mytree.tree_.value)

' polynom --------------------------------------------------------------------'
# numpy.linspace(start, stop, num_of_samples); Returns num evenly spaced samples,
# calculated over the interval [start, stop]
X = np.linspace(-2, 2, num=401)

# concatenate((a1, a2, ...), axis=0); combines NumPy arrays together
# axis=None => arrays are flattened before use.
y1 = np.zeros(200)
y2 = np.ones(1)
y3 = np.ones(200)
Y = np.concatenate((y1, y2, y3), axis=None)
# weights functions
w1 = np.concatenate(((2 / 7) * (np.ones(150)), np.zeros(101)), axis=None)

# 7/2 is godel of range [0.25,2] + [-2,-0.25]
# the first 150 values ([-2,0.25]) have a weight of 2/7
# the next 101 values ([-0.25,0.25]) have a weight of 0
# the last 150 values ([0.25,2]) have a weight of 2/7
# weight = np.concatenate((partW1, (2/7)*np.ones(75)), axis=None)
w2 = (2 / 7) * np.ones(150)
weight = np.concatenate((w1, w2), axis=None)
# we use the above function in polyfit to help calculate the polynome
# polyfit(x, y, deg, w); least square polynomial fit
pf = np.polyfit(X, Y, 32, w=weight)
# print(pf)
# poly1d(c_or_r); The polynomial’s coefficients in decreasing powers
p = np.poly1d(pf)
# print(p)
myline = np.linspace(-2, 2, 401)
# plt.scatter(X, Y)
# plt.plot(myline, p(myline))
# plt.show()
' --------------------------------------------------------------------------'

# leaves = list(leaves)


# def Tree_Predict(v, x):
#     if v in leaves:
#         for i in range(len(leaves)):
#             if (leaves_value_list[i].all() == (mytree.tree.tree_.value[
#                 leaves[i]]).all()):  # checs node's index in original tree and returns one hot encoding of this value
#                 return mytree.leaf_value[i]
#
#     return Tree_Predict(v, x)


# 'v= mytree, x= X_train'
# # Tree_Predict(mytree,X_train)
# print(leaves)
#
# leaves = list(leaves)
# for i in range(len(leaves)):
#     if (leaves_value_list[i].all() == (mytree.tree.tree_.value[leaves[i]]).all()):
#         print(mytree.leaf_value[i])

# def Tree_Predict(v):
#     n_nodes = v.tree_.node_count
#     children_left = v.tree_.children_left
#     children_right = v.tree_.children_right
#     feature = v.tree_.feature
#     threshold = v.tree_.threshold
#
#     node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
#     is_leaves = np.zeros(shape=n_nodes, dtype=bool)
#     stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
#     while len(stack) > 0:
#         # `pop` ensures each node is only visited once
#         node_id, depth = stack.pop()
#         node_depth[node_id] = depth
#
#         # If the left and right child of a node is not the same we have a split
#         # node
#         is_split_node = children_left[node_id] != children_right[node_id]
#         # If a split node, append left and right children and depth to `stack`
#         # so we can loop through them
#         if is_split_node:
#             stack.append((children_left[node_id], depth + 1))
#             stack.append((children_right[node_id], depth + 1))
#         else:
#             is_leaves[node_id] = True
#         print(stack)
#     print(node_id)
#
#
# Tree_Predict(mytree.tree)

fn = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
cn = ['setosa', 'versicolor', 'virginica']

# Setting dpi = 300 to make image clearer than default
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=300)
# fig = plt.figure(figsize=(25,20))
fig, axes = plt.subplots(figsize=(4, 1), dpi=300)
tree.plot_tree(Dtree,
               feature_names=fn,
               class_names=cn,
               filled=True,
               )
# plt.show()


DT = builtTree(Dtree)

# DT.getNode()
printTree(DT)

# data = [0.61, -0.16, 0.62, 0.25]
# sample_data1 = int(Dtree.predict([[0.61, -0.16, 0.62, 0.25]]))
# print(iris.target_names[sample_data1])
data = [-0.333333, -0.80000, 0.9, -0.073333]
sample_data1 = int(Dtree.predict([data]))
print(iris.target_names[sample_data1])
print('new fun')


def Tree_Predict(DT, depth=0):
    if DT is None:
        return

    feature, threshold, leaf, left, right = DT.getNode()

    if isinstance(leaf, np.ndarray):
        # print(depth * ' ', 'leaf = ', leaf)
        return leaf
    else:
        print('phi1=', p(data[feature] - threshold))
        print('phi2=', p(threshold - data[feature]))
        return (p(data[feature] - threshold)) * Tree_Predict(right, depth + 4) + (
            p(threshold - data[feature])) * Tree_Predict(left, depth + 4)


print(Tree_Predict(DT))
