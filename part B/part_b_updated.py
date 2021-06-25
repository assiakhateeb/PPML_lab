import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from my_tree import *


def load_data(data_set):
    if data_set == 'iris':
        data_set = load_iris()
    X = pd.DataFrame(data_set.data[:, :], columns=data_set.feature_names[:])
    Y = pd.DataFrame(data_set.target, columns=["Species"])
    return X, Y, data_set


def scaling(X, data_set):
    """ preprocessing 1 - Feature scaling """
    # rescale a range between an arbitrary set of values [a, b] where a=-1, b=1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_rescaled_features = scaler.fit_transform(X)
    X_rescaled_features = pd.DataFrame(X_rescaled_features[:, :], columns=data_set.feature_names[:])
    return X_rescaled_features


def make_decision_tree(X_rescaled_features, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X_rescaled_features, Y, test_size=0.25)
    ' Now lets fit a DecisionTreeClassifier instance '
    d_tree = DecisionTreeClassifier(max_depth=2)
    d_tree.fit(X_train, Y_train)
    return d_tree, X_test, Y_test


def sklearn_score(d_tree, X_test, Y_test):
    'calculate the score'
    s = d_tree.score(X_test, Y_test)
    print("sklearn_score =", s)
    return s


def sklearn_prediction(d_tree, pred_vec, data_set_name):
    'Making a Prediction on a new sample'
    d = data_set_name
    sample_data1 = int(d_tree.predict([pred_vec]))
    print(d.target_names[sample_data1])


def print_tree(d_tree):
    fn = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    cn = ['setosa', 'versicolor', 'virginica']

    # Setting dpi = 300 to make image clearer than default
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=300)
    # fig = plt.figure(figsize=(25,20))
    fig, axes = plt.subplots(figsize=(4, 1), dpi=300)
    tree.plot_tree(d_tree,
                   feature_names=fn,
                   class_names=cn,
                   filled=True,
                   )
    plt.show()


def polynom(degree, window):
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
    w1 = np.concatenate((window * (np.ones(150)), np.zeros(101)), axis=None)

    # 7/2 is godel of range [0.25,2] + [-2,-0.25]
    # the first 150 values ([-2,0.25]) have a weight of 2/7
    # the next 101 values ([-0.25,0.25]) have a weight of 0
    # the last 150 values ([0.25,2]) have a weight of 2/7
    # weight = np.concatenate((partW1, (2/7)*np.ones(75)), axis=None)
    w2 = window * np.ones(150)
    weight = np.concatenate((w1, w2), axis=None)
    # we use the above function in polyfit to help calculate the polynome
    # polyfit(x, y, deg, w); least square polynomial fit
    pf = np.polyfit(X, Y, degree, w=weight)
    # print(pf)
    # poly1d(c_or_r); The polynomial’s coefficients in decreasing powers
    phi = np.poly1d(pf)
    # print(phi)
    myline = np.linspace(-2, 2, 401)
    # plt.scatter(X, Y)
    # plt.plot(myline, p(myline))
    # plt.show()
    return phi


def Tree_Predict(T, x, phi):
    if T is None:
        return

    feature, threshold, leaf, left, right = T.getNode()
    if isinstance(leaf, np.ndarray):
        return leaf
    else:
        return (phi(x[feature] - threshold)) * Tree_Predict(right, x, phi) + (
            phi(threshold - x[feature])) * Tree_Predict(left, x, phi)


def predict(T, x, phi):
    pred_vec = Tree_Predict(T, x, phi)
    print('predict vec before 1Hot_encoding is:', pred_vec)
    lenHot = len(pred_vec)
    l = np.argmax(pred_vec)
    leaf = np.zeros(lenHot)
    leaf[l] = 1
    print('predict vector with Algorithm 1 is:', leaf)
    return leaf


def pre_processing():
    'load_data'
    data_type = 'iris'
    X, Y, data_set = load_data(data_type)
    '-------pre_processing 1 - Scaling-------'
    X_rescaled_features = scaling(X, data_set)
    d_tree, X_test, Y_test = make_decision_tree(X_rescaled_features, Y)
    '-------calc_Score-------'
    sklearn_score(d_tree, X_test, Y_test)
    '-------predict with sklearn-------'
    # pred_vec = [5, 5, 2.6, 1.5]
    # sklearn_prediction(d_tree, pred_vec, data_set)
    '-------pre_processing 2 - Polynom-------'
    deg = 34
    win = 2/7
    phi = polynom(deg, win)
    '-------pre_processing 3 - make our tree -------'
    myTree = builtTree(d_tree)
    return myTree, phi


def main():
    myTree, phi = pre_processing()
    print('\n', '--------print tree--------')
    printTree(myTree)
    print('\n', '--------now prediction--------')
    data = [0.333333, 0.50000, 0.9, -0.173333]
    predict(myTree, data, phi)


main()
