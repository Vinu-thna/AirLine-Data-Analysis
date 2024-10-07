# importing necessary 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #Imports the function to split datasets into training and testing sets.
from sklearn.linear_model import LinearRegression #Imports the Linear Regression model.
from sklearn.metrics import r2_score, mean_squared_error #Imports metrics for evaluating model performance.
from sklearn.ensemble import RandomForestRegressor #Imports the Random Forest Regressor model.
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

data = pd.read_csv('Airline_Delay.csv')
data.head()
data.info()
data.shape
plt.subplot(2,2,1)
data.describe().T
data.nunique()
data.isnull().sum()

numerical_columns = ['arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct', 'arr_cancelled', 'arr_diverted', 'arr_delay', 'carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())

data.isnull().sum()

#Performance analysis on average delay by carrier
carrier_delay_avg = data.groupby('carrier_name')['arr_delay'].mean().sort_values()
plt.figure(figsize=(10,8))
sns.barplot(x=carrier_delay_avg, y=carrier_delay_avg.index)
plt.title('Average arrival delay by carrier')
plt.xlabel('Average delay (minutes)')
plt.ylabel('Carrier')
plt.show()

#Trend identification for total delays over time
data['date']=pd.to_datetime(data[['year','month']].assign(DAY=1))
monthly_delays = data.groupby('date')['arr_del15'].sum()
plt.figure(figsize=(12,6))
monthly_delays.plot(kind='line')
plt.title('Monthly total delay over time')
plt.xlabel('Date')
plt.ylabel('Total delays')
plt.show()

#Root cause contributing to flight delays
delay_factors=data[['weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct', 'carrier_ct']].mean()
delay_factors.plot(kind='bar')
plt.title('Average contributions to delays')
plt.xlabel('Delay causes')
plt.ylabel('Average count')
plt.xticks(rotation=45)
plt.show()

#converting categorical columns to one encoding
catergorical_columns=['carrier', 'carrier_name', 'airport', 'airport_name']
data = pd.get_dummies(data, columns=catergorical_columns, dtype=int)
data.isnull().sum()

#Data preprocessing for predictive modeling
columns_to_drop = ['year', 'month', 'arr_del15']
data_features = data.drop(columns=columns_to_drop)

X = data_features
datetime_columns = X.select_dtypes(include=['datetime64[ns]']).columns
X = X.drop(columns=datetime_columns)

X = X.astype(float)
y = data['arr_del15'] #taget variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("LINEAR REGRESSION RESULT")
print(f" R^2 score: {r2_score(y_test, y_pred_lr)}")
print(f" MSE: {mean_squared_error(y_test, y_pred_lr)}")

#Random forest regression model
data_cleaned = data.dropna(subset=['arr_delay'])

features = ['arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'nas_ct', 'late_aircraft_ct', 'arr_cancelled', 'arr_diverted']
target = 'arr_delay'

X = data_cleaned[features]
y = data_cleaned[target]

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('')
print("RANDOM FOREST REGRESSION MODEL")
print(" Training score:" ,model.score(X_train, y_train))
print(" Testing score:", model.score(X_test, y_test))
