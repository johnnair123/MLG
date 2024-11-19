import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df=pd.read_csv('https://raw.githubusercontent.com/safal/DS-
ML/main/house_price.csv')

print(df.head())
df.replace(to_replace={'yes':1,'no':0,'unfurnished':0,'furnished':1,'semi-furnished':2},
inplace=True)
print(df.head())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
y_train_reshape=y_train.values.reshape(-1,1)
y_test_reshape=y_test.values.reshape(-1,1)
model = LinearRegression()
model.fit(x_train, y_train_reshape)
print("Intercept:", intercept)
for i,j in zip(x.columns, model.coef_[0]):
print(f"{i} : {j}")
y_pred_test=model.predict(x_test)
y_pred_train=model.predict(x_train) print("r2_score:",r2_score(y_test,
y_pred_test)) print("mean_squared_error",mean_squared_error(y_test,
y_pred_test))
print("mean_absolute_error",mean_absolute_error(y_test, y_pred_test))
