from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load the csv file into a DataFrame from github
df = pd.read_csv("https://github.com/fayez-max/Real-Estate-Data-Set/raw/main/Real%20estate%20valuation%20data%20set.csv")

# Remove rows with missing values
df = df.dropna()

# Check for and remove duplicate rows
df = df.drop_duplicates()

# select features with the strongest correlation
X = df[['X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]

# select the target variable
y = df['Y house price of unit area']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
