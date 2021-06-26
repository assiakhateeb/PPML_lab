from part_b import *
import sys


def main(argv):
    data_type = str(input("Please select a data type(iris/wine): "))
    print('data_type =', data_type)
    myTree, phi, X_test, Y_test, sklearn_tree = pre_processing(data_type)
    print('\n', '--------print tree--------')
    printTree(myTree)
    print('\n', '--------now prediction--------')

    calc_algorithm1_accuracy(myTree, phi, X_test, Y_test)
    sklearn_score(sklearn_tree, X_test, Y_test)
    print_tree(sklearn_tree, data_type)


if __name__ == '__main__':
    main(sys.argv[1:])
