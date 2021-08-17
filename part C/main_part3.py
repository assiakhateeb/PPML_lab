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

            print('phi=', phi)
            print('p=', p)
            print('X_test len', len(X_test), "\nY_test", len(Y_test), '\n', 'X_test=', X_test, 'Y_tset=', Y_test)

            X_test_encrypted = trainset_enc(X_test,context)
            print("Reeeeeeem")

            print('X_test encrypted = ', X_test_encrypted)

            for i in range(len(X_test_encrypted)):
                decrypted_X_test = X_test_encrypted[i].decrypt(sk).tolist()
                print(decrypted_X_test)

            print('************Tree Predict*************')
            result = []
            for i in range(len(X_test_encrypted)):
                startTime = time.time()
                predict = Tree_Predict_partC(myTree, X_test_encrypted[i], phi)
                print("run time =%.4f" % (time.time() - startTime), "seconds")
                print('predict=', predict.decrypt(sk).tolist())
                predict = predict.decrypt(sk).tolist()
                print('Y=', Y_test[i])
                result.append(predict)


            'one hot encoding to the prediction'
            one_hot_result = []
            for res in result:
                print(res)
                one_hot_result.append(one_hot_encoding(res))
            print('one=', one_hot_result)
            accuracy = calc_accuracy_partC(one_hot_result, Y_test)
            print('accuracy=', accuracy)
            # predict = Tree_Predict_partC(myTree, X_test_encrypted[0], phi)
            # print(predict.decrypt(sk).tolist())
            # print(Y_test[0])

            # calc_accuracy_partC(myTree, phi, X_test_encrypted, Y_test)

        except:
            print("UnExpected Error: ", sys.exc_info()[0])
        print('\n')
        select_data_type = input("Please select a data type(iris/wine/cancer) or type exit to stop: ")


folder = 'tree images'
# os.makedirs(folder)
main(sys.argv[1:])
