import tenseal as ts
import part_b as partB
import numpy as np

def pre_processing_partC(data_type):
    'load_data'
    X, Y, data_set = partB.load_data(data_type)
    '-------pre_processing 1 - Scaling-------'
    X_rescaled_features = partB.scaling(X)
    scikit_learn_tree, X_test, Y_test = partB.make_decision_tree(X_rescaled_features, Y)

    '-------pre_processing 2 - Polynom-------'

    phi = partB.polynom(8, 2 / 7)

    '-------pre_processing 3 - make our tree -------'
    myTree = partB.builtTree(scikit_learn_tree)
    return myTree, phi, X_test, Y_test, scikit_learn_tree


def Tree_Predict_partC(T, x, phi):
    if T is None:
        return
    feature, threshold, leaf, left, right = T.getNode()
    # print('x=', x, 'x type=', type(x))
    # print('x decrypted', x.decrypt(sk).tolist())
    # print(x[0], x[0].decrypt(sk).tolist())
    feature = int(abs(feature))
    # print('feature=', feature)
    # print(x[feature].decrypt(sk).tolist())
    if isinstance(leaf, np.ndarray):
        return leaf
    else:
        return (x[feature] - threshold).polyval(phi) * Tree_Predict_partC(right, x, phi) + (
            (threshold - x[feature])).polyval(phi) * Tree_Predict_partC(left, x, phi)


def one_hot_encoding(pred_vec):
    print(pred_vec)
    lenHot = len(pred_vec)
    l = np.argmax(pred_vec)
    leaf = np.zeros(lenHot)
    leaf[l] = 1
    return leaf


def calc_accuracy_partC(res_vec, Y_test):
    counter = 0
    if len(res_vec[0]) == 3:
        for i in range(len(res_vec)):
            if np.logical_and(res_vec[i] == [1, 0, 0], Y_test[i] == 0).all() or np.logical_and(res_vec[i] == [0, 1, 0],Y_test[i] == 1).all() or np.logical_and(res_vec[i] == [0, 0, 1], Y_test[i] == 2).all():
                counter += 1
    if len(res_vec[0]) == 2:
        for i in range(len(res_vec)):
            if np.logical_and(res_vec[i] == [1, 0], Y_test[i] == 0).all() or np.logical_and(res_vec[i] == [0, 1], Y_test[i] == 1).all():
                counter += 1
    return (counter / len(res_vec)) * 100


def trainset_enc(X_test,context):
    pk = context.copy()
    encrypted_Xtest = []
    for vec in X_test:
        print(vec)
        encrypted_Xtest.append(ts.ckks_tensor(pk, vec))
    return encrypted_Xtest





