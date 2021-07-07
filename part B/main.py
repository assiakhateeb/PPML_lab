from part_b import *
import sys
import os


def main(argv):
    select_data_type = input("Please select a data type(iris/wine/cancer) or type exit to stop: ")
    data_type = str(select_data_type)
    print('data_type =', data_type)
    while select_data_type.lower() != 'exit':
        try:
            myTree, phi1, phi2, phi3, phi4, X_test, Y_test, scikit_learn_tree = pre_processing(data_type)
            print_tree(scikit_learn_tree, data_type)
            print('\n', '--------print tree--------')
            printTree(myTree)
            print('\n')
            check_best_accuracy(myTree, phi1, phi2, phi3, phi4, X_test, Y_test, scikit_learn_tree)

        except:
            print("UnExpected Error: ", sys.exc_info()[0])
        print('\n')
        select_data_type = input("Please select a data type(iris/wine/cancer) or type exit to stop: ")


folder = 'tree images'
os.makedirs(folder)
main(sys.argv[1:])
