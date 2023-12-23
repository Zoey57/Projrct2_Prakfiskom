# Importing necessary libraries
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the dataset
FileDB = 'cos.txt'
Database = pd.read_csv(FileDB, sep=",", header=0)

# Print column names to verify correct names
print("Column Names:", Database.columns)

# Assuming 'Feature' is the correct column name, adjust it if needed
feature_column = 'Feature'

# Extracting features (x) and target (y)
x = Database[[feature_column]]
y = Database['Target']

# Fit Decision Tree model
reg = DecisionTreeRegressor(random_state=0)
reg.fit(x, y)

# Generate predictions
xx = np.arange(1, 21, 1)
n = len(xx)
print("----------------------------")
print("-Decision Tree Predictions-")
print("xx(i) Decision Tree")
for i in range(n):
    y_prediction = reg.predict([[xx[i]]])
    print('{:.2f}'.format(xx[i]), y_prediction)
print("----------------------------")

# Plotting predictions
y_dct2 = reg.predict(x)
plt.figure()
plt.plot(x, y_dct2, color='red')
plt.scatter(x, y, color='blue')
plt.title('Prediction Using Decision Tree')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Decision Tree', 'Data'], loc='upper right')
plt.show()
