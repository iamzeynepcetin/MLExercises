import pandas as pd

ad = pd.read_csv('Advertising.csv',delimiter=(','),usecols= range(1,5))
df = ad.copy()

'''df.info()
RangeIndex: 200 entries, 0 to 199
Data columns (total 4 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   TV         200 non-null    float64
 1   radio      200 non-null    float64
 2   newspaper  200 non-null    float64
 3   sales      200 non-null    float64
dtypes: float64(4)
memory usage: 6.4 KB
'''

'''df.describe().T
          count      mean        std  min     25%     50%      75%    max
TV         200.0  147.0425  85.854236  0.7  74.375  149.75  218.825  296.4
radio      200.0   23.2640  14.846809  0.0   9.975   22.90   36.525   49.6
newspaper  200.0   30.5540  21.778621  0.3  12.750   25.75   45.100  114.0
sales      200.0   14.0225   5.217457  1.6  10.375   12.90   17.400   27.0
'''

'''df.isnull().values.any()
False
'''

'''df.corr()
                 TV     radio  newspaper     sales
TV         1.000000  0.054809   0.056648  0.782224
radio      0.054809  1.000000   0.354104  0.576223
newspaper  0.056648  0.354104   1.000000  0.228299
sales      0.782224  0.576223   0.228299  1.000000
'''
'''import seaborn as sns 
sns.pairplot(df, kind = 'reg')
'''
#Plotu inceledigimizde sales ve Tv arasinda duzenli bir iliski oldugunu gorduk

X = df[['TV']]

'''X[0:5]
Out[39]: 
      TV
0  230.1
1   44.5
2   17.2
3  151.5
4  180.8
'''

import statsmodels.api as sm 
X= sm.add_constant(X)

'''X[0:5]
Out[42]: 
   const     TV
0    1.0  230.1
1    1.0   44.5
2    1.0   17.2
3    1.0  151.5
4    1.0  180.8
'''

y = df['sales']
lm = sm.OLS(y,X)
model = lm.fit()
'''model.summary()
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  sales   R-squared:                       0.612
Model:                            OLS   Adj. R-squared:                  0.610
Method:                 Least Squares   F-statistic:                     312.1
Date:                Sun, 04 Jul 2021   Prob (F-statistic):           1.47e-42
Time:                        13:13:07   Log-Likelihood:                -519.05
No. Observations:                 200   AIC:                             1042.
Df Residuals:                     198   BIC:                             1049.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          7.0326      0.458     15.360      0.000       6.130       7.935
TV             0.0475      0.003     17.668      0.000       0.042       0.053
==============================================================================
Omnibus:                        0.531   Durbin-Watson:                   1.935
Prob(Omnibus):                  0.767   Jarque-Bera (JB):                0.669
Skew:                          -0.089   Prob(JB):                        0.716
Kurtosis:                       2.779   Cond. No.                         338.
==============================================================================
'''
#buradaki ciktiyi modeli yorumlamak icin kullaniyoruz coef katsayisi B0 ve B1 katsayilarini temsil eder
#B0 TV olmadan gerceklesen satis katsayisi TV de 1 birim artis oldugunda satista B1 kadar artis olmaktadir. 

'''print(model.params)
const    7.032594
TV       0.047537
dtype: float64
'''
#katsayilara eristik
 
print('Sales=' + str('%.4f' % model.params[0]) + ' + TV' + '*' + str('%.4f ' % model.params[1]))
#Sales=7.0326 + TV*0.0475 
 
#model.predict(X)[30, 40, 50]

































