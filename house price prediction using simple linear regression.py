
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\fsds materials\fsds\3. Aug\11th\SLR - Practicle\House_data.csv')

space = dataset['sqft_living']

price = dataset['price']

x = np.array(space).reshape(-1,1)
y = np.array(price)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =1/3, random_state =0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

%matplotlib inline
plt.scatter(x_train, y_train, color ='red')
plt.plot(x_train, regressor.predict(x_train),color ='blue')
plt.title("Visuals for Test Dataset")
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()

plt.scatter(x_test, y_test, color ='red')
plt.plot(x_train, regressor.predict(x_train),color ='blue')
plt.title('Visuals of Test Dataset')
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()
