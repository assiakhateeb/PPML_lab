import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from time import time
from my_tree import *


def load_data(data_set):
    if data_set == 'iris':
        data_set = load_iris()
    if data_set == 'wine':
        data_set = load_wine()
    if data_set == 'cancer':
        data_set = load_breast_cancer()
    # X = pd.DataFrame(data_set.data[:, :], columns=data_set.feature_names[:])
    # Y = pd.DataFrame(data_set.target, columns=["Species"])
    X = data_set.data[:, :]
    Y = data_set.target
    return X, Y, data_set


def scaling(X):
    """ preprocessing 1 - Feature scaling """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_rescaled_features = scaler.fit_transform(X)
    return X_rescaled_features


def make_decision_tree(X_rescaled_features, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X_rescaled_features, Y, test_size=0.35)
    ' Now lets fit a DecisionTreeClassifier instance '
    max_depth = 4
    d_tree = DecisionTreeClassifier(max_depth=4)
    print('max depth =', max_depth)
    d_tree.fit(X_train, Y_train)
    return d_tree, X_test, Y_test


def scikit_learn_score(scikit_learn_tree, X_test, Y_test):
    'calculate the score'
    s = scikit_learn_tree.score(X_test, Y_test)*100
    print('scikit_learn_score =%.5f' % s, '%')
    return s


def scikit_learn_prediction(d_tree, pred_vec, data_set_name):
    'Making a Prediction on a new sample'
    d = data_set_name
    sample_data1 = int(d_tree.predict([pred_vec]))
    print('scikit_learn_prediction =', d.target_names[sample_data1])


def print_tree(scikit_learn_tree, data_type):
    if data_type == 'iris':
        feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        target_names = ['setosa', 'versicolor', 'virginica']
    if data_type == 'wine':
        feature_names = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids',
              'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines',
              'proline']
        target_names = ['class_0', 'class_1', 'class_2']
    if data_type == 'cancer':
        feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error',
        'fractal dimension error', 'worst radius', 'worst texture',
        'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points',
        'worst symmetry', 'worst fractal dimension']
        target_names = ['malignant', 'benign']
    # Setting dpi = 300 to make image clearer than default
    fig, axes = plt.subplots(figsize=(4, 2), dpi=300)
    tree.plot_tree(scikit_learn_tree,
                   feature_names=feature_names,
                   class_names=target_names,
                   filled=True,
                   )
    fig.savefig('tree images/' + data_type + ' tree.jpg')
    plt.show()


def polynom(degree, window):
    # numpy.linspace(start, stop, num_of_samples); Returns num evenly spaced samples,
    # calculated over the interval [start, stop]
    X = np.linspace(-2, 2, num=201)
    y1 = np.zeros(100)
    y2 = np.ones(1)
    y3 = np.ones(100)
    Y = np.concatenate((y1, y2, y3), axis=None)
    # weights functions
    w1 = np.concatenate((window * (np.ones(75)), np.zeros(51)), axis=None)
    w2 = window * np.ones(75)
    weight = np.concatenate((w1, w2), axis=None)
    # we use the above function in polyfit to help calculate the polynome
    # polyfit(x, y, deg, w); least square polynomial fit
    pf = np.polyfit(X, Y, degree, w=weight)
    phi = np.poly1d(pf)
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


def algorithm1_predict(T, x, phi):
    pred_vec = Tree_Predict(T, x, phi)
    lenHot = len(pred_vec)
    l = np.argmax(pred_vec)
    leaf = np.zeros(lenHot)
    leaf[l] = 1
    return leaf


def pre_processing(data_type):
    'load_data'
    X, Y, data_set = load_data(data_type)
    '-------pre_processing 1 - Scaling-------'
    X_rescaled_features = scaling(X)
    scikit_learn_tree, X_test, Y_test = make_decision_tree(X_rescaled_features, Y)

    '-------pre_processing 2 - Polynom-------'
    phi1 = []
    phi2 = []
    phi3 = []
    phi4 = []
    k = [1/4, 2/7, 1/3, 4/10]
    for deg in range(34):
        phi1.append(polynom(deg, k[0]))
        phi2.append(polynom(deg, k[1]))
        phi3.append(polynom(deg, k[2]))
        phi4.append(polynom(deg, k[3]))

    '-------pre_processing 3 - make our tree -------'
    myTree = builtTree(scikit_learn_tree)
    return myTree, phi1, phi2, phi3, phi4, X_test, Y_test, scikit_learn_tree


def calc_algorithm1_accuracy(myTree, phi, X_test, Y_test):
    res_vec = []
    counter, counter2 = 0, 0
    # startTime = time()
    for x in X_test:
        res = algorithm1_predict(myTree, x, phi)
        res_vec.append(res)
    # print("run time =%.4f" % (time() - startTime), "seconds")
    if len(res_vec[0]) == 3:
        for i in range(len(res_vec)):
            if np.logical_and(res_vec[i] == [1, 0, 0], Y_test[i] == 0).all() or np.logical_and(res_vec[i] == [0, 1, 0], Y_test[i] == 1).all() or np.logical_and(res_vec[i] == [0, 0, 1], Y_test[i] == 2).all():
                counter += 1
    if len(res_vec[0]) == 2:
        for i in range(len(res_vec)):
            if np.logical_and(res_vec[i] == [1, 0], Y_test[i] == 0).all() or np.logical_and(res_vec[i] == [0, 1], Y_test[i] == 1).all():
                counter += 1
    algorithm1_score = (counter / len(res_vec))*100
    # print('algorithm1_score =%.4f' % algorithm1_score, '%')
    return algorithm1_score


def check_best_accuracy(myTree, phi1, phi2, phi3, phi4, X_test, Y_test, scikit_learn_tree ):

    scikit_learn_score(scikit_learn_tree, X_test, Y_test)
    phi = [phi1, phi2, phi3, phi4]
    for p in phi:
        acc = []
        # startTime = time()
        for poly in p:
            acc.append(calc_algorithm1_accuracy(myTree, poly, X_test, Y_test))
        best_acc = max(acc)
        # if p == phi1:
        #     print('Best Accuracy For Algorithm 1 = %.4f' % best_acc, 'With Polynom degree =', np.argmax(acc),
        #           'And Window = 0', ",Run time =%.4f" % (time() - startTime), "sec")
        # if p == phi2:
        #     print('Best Accuracy For Algorithm 1 = %.4f' % best_acc, 'With Polynom degree =', np.argmax(acc),
        #           'And Window = 0.25', ",Run time =%.4f" % (time() - startTime), "sec")
        # if p == phi3:
        #     print('Best Accuracy For Algorithm 1 = %.4f' % best_acc, 'With Polynom degree =', np.argmax(acc),
        #           'And Window = 0.5', ",Run time =%.4f" % (time() - startTime), "sec")
        # if p == phi4:
        #     print('Best Accuracy For Algorithm 1 = %.4f' % best_acc, 'With Polynom degree =', np.argmax(acc),
        #           'And Window = 0.75', ",Run time =%.4f" % (time() - startTime), "sec")
    print('Best Accuracy For Algorithm 1 = %.4f' % best_acc, '%','With Polynom degree =', np.argmax(acc))


