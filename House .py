import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# reading the main data 

df=pd.read_csv("F:\DATA\/House.csv")


# cleaning the data 
# address 
df = df[df['Address'].notna() & (df['Address'] != "")]
# Area
df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
df = df[df['Area'] <= 900]

#____________________________________________________________

# turning bollean datas to numbers 

df['Parking'] = df['Parking'].astype(int)
df['Warehouse'] = df['Warehouse'].astype(int)
df['Elevator'] = df['Elevator'].astype(int)

# normalize the address

df = pd.get_dummies(df, columns=['Address'], drop_first=True)
# area,room,address,parking,warehouse,elavotor are correct    
df.to_csv('House_cleaned.csv', index=False)
#__________________________________________________________

x = df.drop(['Price', 'Price(USD)'], axis=1)  
y = df['Price(USD)']  

# normalize the data

x = x / x.max()
y = y / y.max()

# devide the data to 90% for train and 10% for the test

msk = np.random.rand(len(df)) < 0.9
train_x = x[msk]
test_x = x[~msk]
train_y = y[msk]
test_y = y[~msk]


# random forrest

regr = RandomForestRegressor(n_estimators=100, random_state=42)          # 100 , 42 are Usual
regr.fit(train_x, train_y)


# test the data

test_y_ = regr.predict(test_x)

MAE = mean_absolute_error(test_y, test_y_)
MSE=np.mean((test_y_ - test_y)**2)
r2 = r2_score(test_y, test_y_)

print('Mean Absolute Error:',MAE)
print('MSE:',MSE)
print('R2:',r2)



