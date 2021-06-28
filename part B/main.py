from part_b import *
import sys


def main(argv):
    select_data_type = input("Please select a data type(iris/wine/cancer) or type exit to stop: ")
    while select_data_type.lower() != 'exit':
        try:
            data_type = str(select_data_type)
            print('data_type =', data_type)
            myTree, phi, X_test, Y_test, sklearn_tree = pre_processing(data_type)
            print('\n', '--------print tree--------')
            printTree(myTree)
            print('\n', '--------now prediction--------')

            calc_algorithm1_accuracy(myTree, phi, X_test, Y_test)
            sklearn_score(sklearn_tree, X_test, Y_test)
            print_tree(sklearn_tree, data_type)

        except:
            print("UnExpected Error: ", sys.exc_info()[0])
        select_data_type = input("Please select a data type(iris/wine/cancer) or type exit to stop: ")


if __name__ == '__main__':
    main(sys.argv[1:])
