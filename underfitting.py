import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

filepath = './Datasets/train.csv'

filedata = pd.read_csv(filepath)

filedata = filedata.fillna(0)

y = filedata.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

x = filedata[features]

# Splitting the data into test and training sets
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)

# Utility function to compare MAE scores from different values
def get_mae(max_leaf_nodes, training_x, valid_x, training_y, valid_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(training_x, training_y)
    predictions = model.predict(valid_x)
    final_mae = mean_absolute_error(valid_y, predictions)
    return final_mae

for max_leaf_nodes in [5, 50, 500, 5000]:
    cal_mae = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)
    print(cal_mae)

candidate_max_leaf_nodes = []
scores = {leaf_size: get_mae(leaf_size, train_x, val_x, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)