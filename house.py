import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv(
    r'_YOUR_LOCATION_\house_data.csv')

columns = ['bedrooms', 'bathrooms', 'floors', 'yr_built', 'price']
df = df[columns]

X = df.iloc[:, 0:4]
y = df.iloc[:, 4:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

lr = LinearRegression()
lr.fit(X_train, y_train)

pickle.dump(lr, open('model.pkl', 'wb'))
