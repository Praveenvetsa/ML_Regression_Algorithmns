import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the dataset
dataset = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\fsds materials\fsds\3. Aug\11th\SIMPLE LINEAR REGRESSION\Salary_Data.csv')

# split the data to independent variable

X  = dataset.iloc[:,:-1].values

# split the data to Dependent variable

y = dataset.iloc[:,1].values

# as dependent variable is continous so we use regression algorithm
# as the dataset having two attributes we use the simple linear regression algorithm
# split the dataset to 80-20%

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =0)

# we called simple linear regression algorithm from sklearn framework

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# we build simple linear regression model regressor

regressor.fit(X_train, y_train)

# test the model and create a predicated table

y_pred = regressor.predict(X_test)

# visualize train data point (24 data record)
%matplotlib inline

plt.scatter(X_train, y_train, color ='red')
plt.plot(X_train, regressor.predict(X_train),color ='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('years of Experience')
plt.ylabel("Salary")
plt.show()

# Visualize test data point
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.title("Salary vs Experience (Training set)")
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()

# slope is generated from linear regressor algorithm which fit to dataset
m = regressor.coef_

# intercept also generate by model
c = regressor.intercept_

# predict or forecast the future the data which we not trained before

y_12 = 9312.57 * 12 + 26780
y_12

y_25 = 9312.57 * 25 + 26780
int(y_25)

# to check overfitting (low bias high variance)
bias = regressor.score(X_train, y_train)
bias

# to check underfitting (high bias low variance)
variance = regressor.score(X_test, y_test)
variance
