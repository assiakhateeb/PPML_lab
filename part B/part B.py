# jupyter-lab

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

iris = load_iris()
# print(iris)
X = pd.DataFrame(iris.data[:, :], columns=iris.feature_names[:])
# print(X)
""" preprocessing 1 - Feature scaling """
# rescale a range between an arbitrary set of values [a, b] where a=-1, b=1
scaler = MinMaxScaler(feature_range=(-1, 1))
X_rescaled_features = scaler.fit_transform(X)
X_rescaled_features = pd.DataFrame(X_rescaled_features[:, :], columns=iris.feature_names[:])
print(X_rescaled_features)

Y = pd.DataFrame(iris.target, columns=["Species"])
# print(Y)
""" preprocessing 1 - Feature scaling """
Y_rescaled_features = scaler.fit_transform(Y)
Y_rescaled_features = pd.DataFrame(Y_rescaled_features, columns=["Species"])
print(Y_rescaled_features)

X_train, X_test, Y_train, Y_test = train_test_split(X_rescaled_features, Y_rescaled_features, test_size=0.25)
# Now let's fit a DecisionTreeClassifier instance
Dtree = DecisionTreeClassifier(max_depth=5)
Dtree.fit(X_train, Y_train)

"""calculate the score"""
s = Dtree.score(X_test, Y_test)
print("\n", "score=", s, "\n")

# Visualizing the build Decision tree
# from sklearn.tree import export_graphviz
#
# """ gini is a measure that is used to find a node to split on it
#     samples refer to the number of samples that are entering this node
#     value is a split of the data instances according to their class
#      example: value = [instances_num_of_setosa, instances_num_of_versicolor, instances_num_of_virginica
#     class is what the decision tree would predict
# """
# # Creates dot file named tree.dot
# export_graphviz(Dtree, out_file="myTree.dot", feature_names=list(X.columns),
#                 class_names=iris.target_names,
#                 filled=True, rounded=True)
#
# # convert .dot file to png
# """ https://onlineconvertfree.com/complete/dot-png/ """

' Making a Prediction on a new sample '
sample_data1 = int(Dtree.predict([[5, 5, 1, 3]]))
print(iris.target_names[sample_data1])

sample_data1 = int(Dtree.predict([[5, 5, 2.6, 1.5]]))
print(iris.target_names[sample_data1])

sample_data1 = int(Dtree.predict([[5, 5, 15, 5]]))
print(iris.target_names[sample_data1])

# tree.plot_tree(Dtree)
# plt.show()

# Visualizing the build Decision tree with dot file
#     gini is a measure that is used to find a node to split on it
#     samples refer to the number of samples that are entering this node
#     value is a split of the data instances according to their class
#     value = [instances_num_of_setosa, instances_num_of_versicolor, instances_num_of_virginica]
#     class is what the decision tree would predict

fn = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
cn = ['setosa', 'versicolor', 'virginica']

# Setting dpi = 300 to make image clearer than default
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)

tree.plot_tree(Dtree,
               feature_names=fn,
               class_names=cn,
               filled=True);

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = onehot_encoder.fit_transform(Y_rescaled_features)
print(y_onehot.shape)
print(Y_rescaled_features[0:5], "\n")
print(y_onehot[1:5], "\n")
print(y_onehot[149], "\n")
print(y_onehot[50], "\n")
