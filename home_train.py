import pandas as pd # Reading the Data
from sklearn.tree import DecisionTreeRegressor # Building the model
from sklearn.metrics import mean_absolute_error as mae # Testing the model
from sklearn.model_selection import train_test_split # Splitting the dataset into parts for testing and training

# Save and Read Dataset into a variable
home_datapath = './Datasets/train.csv'

# Using Pandas to read the data in preparation to train the model
home_data = pd.read_csv(home_datapath)

# View list of columns in the Dataset
# print(home_data.columns)

# drop any incomplete or empty columns
home_data = home_data.fillna(0)

# Identify the label of the data(value we want to predict later)
y = home_data.SalePrice

# Identify the columns we want to train the data on
home_features = ['LotArea', 'YearBuilt', "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

# Select the data according to selected features
x = home_data[home_features]

# Splitting the data into training and validation data for both features and target
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)

# Define the model and level of randomness
home_model = DecisionTreeRegressor(random_state=1)

# Fit the model
home_model.fit(train_x, train_y)

# Run predictions on the Model(with validation dataset) / get predicted prices
predictions = home_model.predict(val_x)

# Check the loss function(difference between actual values and predicted values)
#loss_value = mae(y, predictions)

# Check the loss functions for validation split and predicted values
val_mae = mae(val_y, predictions)
print("Mean Absolute Error:", val_mae)