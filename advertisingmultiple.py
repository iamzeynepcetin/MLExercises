import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression


ad = pd.read_csv('Advertising.csv',delimiter=(','), usecols= range(1,5))
df = ad.copy()

X =df[df.columns[0:3]]
y = df["sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 42)

lm = LinearRegression()
model = lm.fit(X_train, y_train)

model.intercept_ #sabit katsayi = 2.979067338122629
model.coef_ #bagimsiz degisken katsayilari  = [0.04472952 0.18919505 0.00276111]


x = model.intercept_
y = model.coef_[0]
z = model.coef_[1]
k = model.coef_[1]

print('Sales = {0} + {1}*TV + {2}*radio + {3}*newspaper'.format(x,y,z,k))

'''
Sales = 2.979067338122629 + 0.044729517468716326*TV + 0.18919505423437658*radio + 0.18919505423437658*newspaper
'''

predict_data = [[30], [10], [40]]
predict_data = pd.DataFrame(predict_data).T

model.predict(predict_data)
'''
array([6.32334798]) #Tahmin degeri
'''

