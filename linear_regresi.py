import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import seaborn as sns
df = load_boston()
dff = pd.DataFrame(df.data, columns=df.feature_names)
dff['MDEV'] = df.target
print(dff.head())

# idemtifikasi missing value 
dff.isnull().sum()

# Eksplorasi data

sns.set(rc={'figure.figsize': (11.7 , 8.27)})
sns.distplot(dff['MDEV'], bins=30)
plt.show()
#membuat matrik korelasi untuk melihat hubangan aantara variabel 
correlation_matrix = dff.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()
# data test dan data traning
X = dff[['RM', 'LSTAT']].values
Y = dff['MDEV'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3)

# membuat model
model = LinearRegression()
model.fit(X_train, y_train)

# Intercep dan koefisien
print('\n')
print('Intercept : \n', model.intercept_)
print('\n')
print('Koefisien: \n',model.coef_)

# evaluasi model untuk data training
prediksi = model.predict(X_train)
RMSE = (np.sqrt(mean_squared_error(y_train, prediksi)))
R2 = r2_score(y_train, prediksi)

print('Performa Model Training')
print('-----------------------')
print('RMSE is {}'.format(RMSE))
print('R2 is {}'.format(R2))
print('\n')
#evaluasi model untuk data testing
prediksi1 = model.predict(X_test)
RMSE = (np.sqrt(mean_squared_error(y_test, prediksi1)))
R2 = r2_score(y_test, prediksi1)

print('Performa Model Testing')
print('-----------------------')
print('RMSE is {}'.format(RMSE))
print('R2 is {}'.format(R2))

plt.scatter(y_test, prediksi1)
plt.show()