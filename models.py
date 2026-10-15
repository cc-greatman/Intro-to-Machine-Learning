import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Save filepath to variable for easier access
melbourne_file_path = './Datasets/melb_data.csv'

# Read the data and store data in the DataFrame titled melbourne_data
melbourne_data  = pd.read_csv(melbourne_file_path)

# dropna drop missing values (think of na as not available)
melbourne_data = melbourne_data.dropna(axis=0)

# Selecting the prediction target with dot notation
y = melbourne_data.Price

# Selecting multiple features using the column list method
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = melbourne_data[melbourne_features]

# Define the model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit the model
melbourne_model.fit(x, y)

# Running predictions
print("Making predictions for the following 5 houses:")
print(x.head())
print("The predictions are")
print(melbourne_model.predict(x.head()))

# Print out the list of all the columns in the Dataset
# print(melbourne_data.columns)

# Print a summary of the data in Melbourne Data
# print(x.describe())

# Print the first 5 rows of the Dataset
# print(x.head())