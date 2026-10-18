import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

file_path = './Datasets/train.csv'
file_data = pd.read_csv(file_path)

file_data.fillna(0)

file_features = ['LotArea', 'YearBuilt', "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

y = file_data.SalePrice

x = file_data[file_features]

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)

model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

forest_model = RandomForestRegressor(random_state=1)

model.fit(train_x, train_y)
forest_model.fit(train_x, train_y)

val_pred = model.predict(val_x)
forest_pred = forest_model.predict(val_x)

mae = mean_absolute_error(val_y, val_pred)
forest_mae = mean_absolute_error(val_y, forest_pred)

print(f"Decision MAE = {mae:.2f} && Forest MAE = {forest_mae:.2f}")









