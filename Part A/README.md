# DecisionTree

The project includes implementation of Decision Tree classifier from scratch, without using any machine learning libraries. The Objective of this project is to make prediction and train the model over a dataset (Advertisement dataset, Breast Cancer dataset, Iris dataset). The dataset is split randomly between training and testing set in the ratio of 8:2 respectively. After constructing the decision tree with the training data and applying the appropriate pruning strategy following details are observed in two independent runs:

# Pruning Strategy
To prune each node one by one (except the root and the leaf nodes), and check weather pruning helps in increasing the accuracy, if the accuracy is increased, prune the node which gives the maximum accuracy at the end to construct the final tree (if the accuracy of 100% is achieved by pruning a node, stop the algorithm right there and do not check for further new nodes).

# How to configure
    1. If the system don't have python installed in it, first install any python version (version greater than v2.7).
        https://www.python.org/downloads/
    2. The code has following dependencies, which needs to be installed before running this code: - Pandas. More details at: https://pandas.pydata.org
        from command line: pip install pandas 
        scikit-learn for only one method in the driver code - train test split
        from command line: pip install -U scikit-learn
    3. Open root directory (DecisionTree) of the project and run command from command line: python driver.py
