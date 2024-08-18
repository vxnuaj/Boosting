from gradientboost import GradientBoost
from nue.preprocessing import csv_to_numpy, train_test_split, x_y_split

data = csv_to_numpy("data/DesTreeData.csv")
train, test = train_test_split(data, train_split = .8)
X_train, Y_train = x_y_split(train, y_col = 'last')
X_test, Y_test = x_y_split(test, y_col = 'last')

verbose_train = True
n_trees = 50
alpha = 1
dtree_dict = {
    'verbose_train': False,
    'verbose_test': False,
    'criterion': 'gini',
    'max_depth': 3,
    'min_node_samples': 2,
}

model = GradientBoost(verbose_train = verbose_train)
model.train(X_train, Y_train, n_trees = n_trees, alpha = alpha, dtree_dict = dtree_dict)

