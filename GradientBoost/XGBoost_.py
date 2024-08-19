from xgboost import XGBClassifier
from termcolor import colored
from nue.preprocessing import csv_to_numpy, train_test_split, x_y_split

data = csv_to_numpy('data/data.csv')
train, test = train_test_split(data, train_split = .8)
X_train, Y_train = x_y_split(train, y_col = 'last')
X_test, Y_test = x_y_split(test, y_col = 'last')


model = XGBClassifier(reg_lambda = .1)
print('Training Model')
model.fit(X_train, Y_train)
print('Finished Training')
print('Model Testing')
print(colored(f'\nAccuracy: {model.score(X_test, Y_test.flatten())}', 'green', attrs = ['bold']))



