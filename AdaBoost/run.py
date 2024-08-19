import numpy as np
from adaboost import AdaBoost
from nue.preprocessing import csv_to_numpy, x_y_split, train_test_split

data = csv_to_numpy("data/data.csv")
train, test = train_test_split(data, train_split = .8)
X_train, Y_train = x_y_split(train, y_col = 'last')
X_test, Y_test = x_y_split(test, y_col = 'last')
Y_train = np.where(Y_train == 0, -1, 1)
Y_test = np.where(Y_test == 0, -1, 1)

verbose_train = True
verbose_test = True
n_stumps = 100
criterion = 'entropy'

model = AdaBoost(verbose_train = verbose_train, verbose_test = verbose_test)
model.train(X_train, Y_train, n_stumps = n_stumps, criterion = criterion)
model.test(X_test, Y_test)


