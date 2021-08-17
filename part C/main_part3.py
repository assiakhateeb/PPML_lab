from part_c import *
import sys
import os
import tenseal as ts
import time


def main(argv):
    context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=2 ** 13,
    # coeff_mod_bit_sizes=[40, 21, 21, 21, 21, 21, 21, 40]
    coeff_mod_bit_sizes=[40, 21, 21, 21, 21, 21, 21, 40]
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** 21

    sk = context.secret_key()
    context.make_context_public()

    select_data_type = input("Please select a data type(iris/wine/cancer) or type exit to stop: ")
    data_type = str(select_data_type)
    print('data_type =', data_type)
    while select_data_type.lower() != 'exit':
        try:
            myTree, phi, X_test, Y_test, scikit_learn_tree = pre_processing_partC(data_type)
            p = np.flip(phi)

            print('X_test = ', X_test)
            print('Y_test = ', Y_test)

            X_test_encrypted = trainset_enc(X_test,context)

            print('X_test encrypted = ', X_test_encrypted)


            print('************Tree Predict*************')
            result = []
            for i in range(len(X_test_encrypted)):
                startTime = time.time()
                sample_tree_predict = Tree_Predict_partC(myTree, X_test_encrypted[i], phi)
                print("run time =%.4f" % (time.time() - startTime), "seconds")
                sample_tree_predict = sample_tree_predict.decrypt(sk).tolist()
                print('predict=',sample_tree_predict)
                result.append(sample_tree_predict)

            print('prediction result = ', result)


            'one hot encoding to the prediction'
            one_hot_result = []
            for res in result:
                one_hot_result.append(one_hot_encoding(res))

            print('one hot encoding, prediction result = ', one_hot_result)
            accuracy = calc_accuracy_partC(one_hot_result, Y_test)
            print('prediction accuracy =', accuracy)


        except:
            print("UnExpected Error: ", sys.exc_info()[0])
        print('\n')
        select_data_type = input("Please select a data type(iris/wine/cancer) or type exit to stop: ")


folder = 'tree images'
# os.makedirs(folder)
main(sys.argv[1:])
