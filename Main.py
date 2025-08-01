import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Reading the main data 

df=pd.read_csv("F:\DATA\House\/House.csv")


# Cleaning the data 
 
df = df[df['Address'].notna() & (df['Address'] != "")]
# Area
df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
df = df[df['Area'] <= 900]

# turning bollean datas to numbers 

df['Parking'] = df['Parking'].astype(int)
df['Warehouse'] = df['Warehouse'].astype(int)
df['Elevator'] = df['Elevator'].astype(int)
df = pd.get_dummies(df, columns=['Address'], drop_first=True)

df.to_csv('House_cleaned.csv', index=False)



# Normalize

df = pd.get_dummies(df, columns=['Address'], drop_first=True)


df.to_csv('House_cleaned.csv', index=False)
#__________________________________________________________

# Target x,y 
x = df.drop(['Price', 'Price(USD)'], axis=1)          
y = df['Price(USD)']                                  

# normalize the data

x = x / x.max()                    # or   x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))
y = y / y.max()                 

# Train & test

msk = np.random.rand(len(df)) < 0.9
train_x = x[msk]
test_x = x[~msk]
train_y = y[msk]
test_y = y[~msk]


# Modeling

regr = RandomForestRegressor(n_estimators=100, random_state=42).fit(train_x,train_y)         


# Test

test_y_ = regr.predict(test_x)

MAE = mean_absolute_error(test_y, test_y_)
MSE=np.mean((test_y_ - test_y)**2)
r2 = r2_score(test_y, test_y_)

print('Mean Absolute Error:',MAE)
print('MSE:',MSE)
print('R2:',r2)


