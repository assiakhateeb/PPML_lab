import numpy as np
import tensorflow as tf2
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

# set numpy random seed for reproducibility
np.random.seed(0)

# load train and test data sets

train = np.loadtxt('iris_train.csv', delimiter=',')
test = np.loadtxt('iris_test.csv', delimiter=',')

# split train set into x and y sets
naive_train_x = train[:, :4]
naive_train_y = train[:, 4:]

# split test set into x and y sets
naive_test_x = test[:, :4]
naive_test_y = test[:, 4:]

# build sequential model with an output layer of 3 nodes
naive_model = Sequential()
naive_model.add(Dense(3, input_dim=4, activation='softmax'))

# compile model and fit to train set
naive_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
naive_model.fit(naive_train_x, naive_train_y, epochs=100)

# evaluate on test set and predict classes
naive_metrics = naive_model.evaluate(naive_test_x, naive_test_y)
naive_predictions = naive_model.predict(naive_test_x)


# set numpy random seed for reproducibility
np.random.seed(0)

# scaler to scale x values, fit on train and test data set x values
scaler = MinMaxScaler((-1, 1))
scaler.fit(np.concatenate((train[:, :4], test[:, :4])))

# split train set into x and y sets, scale x set
better_train_x = scaler.transform(train[:, :4])
better_train_y = train[:, 4:]

# split test set into x and y sets, scale x set
better_test_x = scaler.transform(test[:, :4])
better_test_y = test[:, 4:]

# build sequential model with 3 hidden layers of 10 nodes and an output layer of 3 nodes
# batch normalization is run after each layer (and after activation of those layers)
# to normalize the data for faster and better training
better_model = Sequential()
better_model.add(Dense(10, activation='relu'))
better_model.add(BatchNormalization())
better_model.add(Dense(10, activation='relu'))
better_model.add(BatchNormalization())
better_model.add(Dense(3, activation='softmax'))

# compile model and fit to train set
better_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
better_model.fit(better_train_x, better_train_y, epochs=100)

# evaluate on test set and predict classes
better_metrics = better_model.evaluate(better_test_x, better_test_y)
better_predictions = better_model.predict(better_test_x)

# print accuracy and predictions of naive model
print('\nNaive Accuracy: ', naive_metrics[1])
print('\nNaive Predictions:\n', naive_predictions)

# print accuracy and predictions of better model
print('\nBetter Accuracy: ', better_metrics[1])
print('\nBetter Predictions:\n', better_predictions)