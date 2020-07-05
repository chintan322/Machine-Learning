#multiple linear regression

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
ct = ColumnTransformer([("encoder", OneHotEncoder(), [3])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

#Avoiding the Dummy Variable Trap
X = X[:,1:]


#spliting dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# feature scaling

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) 

#predicting the Test set result
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X ,axis = 1)

X_opt = X[:, [0,1,2,3,4,5]]
import statsmodels.regression.linear_model as pq;
X_opt = np.array(X_opt, dtype=float)
regressor_OLS=pq.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

#just eleminate p>|SL| and go on......


X_opt = X[:, [0,3]]
import statsmodels.regression.linear_model as pq;
X_opt = np.array(X_opt, dtype=float)
regressor_OLS=pq.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()




