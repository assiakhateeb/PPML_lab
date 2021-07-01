from part_b import *
import sys
import os


def main(argv):
    select_data_type = input("Please select a data type(iris/wine/cancer) or type exit to stop: ")
    # sys.stdout = open("tree images/results.txt", "w")
    while select_data_type.lower() != 'exit':
        try:
            data_type = str(select_data_type)
            print('data_type =', data_type)
            myTree, phi1, phi2, phi3, phi4, X_test, Y_test, sklearn_tree = pre_processing(data_type)
            print('\n', '--------print tree--------')
            printTree(myTree)
            print('\n', '--------now prediction--------')
            sklearn_score(sklearn_tree, X_test, Y_test)
            print('****************** WINDOW = 0 ******************')
            for p in phi1:
                print('window = 0')
                print('polynom order =', p.order)
                calc_algorithm1_accuracy(myTree, p, X_test, Y_test)
                print('-----------------------------------------')

            # print_tree(sklearn_tree, data_type)

            print('****************** WINDOW = 0.25 ******************')
            for p in phi2:
                print('window = 0.25')
                print('polynom order =', p.order)
                calc_algorithm1_accuracy(myTree, p, X_test, Y_test)
                print('-----------------------------------------')

            # print_tree(sklearn_tree, data_type)

            print('****************** WINDOW = 0.5 ******************')
            for p in phi3:
                print('window = 0.5')
                print('polynom order =', p.order)
                calc_algorithm1_accuracy(myTree, p, X_test, Y_test)
                print('-----------------------------------------')

            # print_tree(sklearn_tree, data_type)
            sklearn_score(sklearn_tree, X_test, Y_test)
            print('****************** WINDOW = 0.75 ******************')
            for p in phi4:
                print('window = 0.75')
                print('polynom order =', p.order)
                calc_algorithm1_accuracy(myTree, p, X_test, Y_test)
                print('-----------------------------------------')

            print_tree(sklearn_tree, data_type)
        except:
            print("UnExpected Error: ", sys.exc_info()[0])
        # sys.stdout.close()
        select_data_type = input("Please select a data type(iris/wine/cancer) or type exit to stop: ")


folder = 'tree images'
os.makedirs(folder)
main(sys.argv[1:])
# sys.stdout = open("results.txt", "w")
