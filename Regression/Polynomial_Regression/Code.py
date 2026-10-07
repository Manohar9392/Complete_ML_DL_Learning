import  pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn .model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import  Pipeline

##-----creating our own  non-linear dataset to perform polynomial regression------------------------
X=6*np.random.rand(100,1)-3       ## rand it creates hundred rows 1 column between vals 0-1
Y=0.5 * X**2 + 1.5*X + 2 + np.random.randn(100,1)       ##  randn it creates hundered rows and 1 column of standard normalize vals consists of positive negative and zero values

## quadratic equation used 0.5X^2+1.5X+2+outliers
## ------------------visualization of non-linear data

# plt.scatter(X,Y,color='red')
# plt.xlabel("x vals")
# plt.ylabel(" y vals")
# plt.title("non linear relationship")
# plt.show()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


##--------implement linear regression model-----------
# regression=LinearRegression()
# regression.fit(X_train,Y_train)
# Y_pred=regression.predict(X_test)
# print(r2_score(Y_test,Y_pred))

##let visualize model
# plt.plot(X_test,regression.predict(X_test) ,color='red')
# plt.scatter(X_train,Y_train)
# plt.xlabel("x dataset")
# plt.ylabel("y dataset")
# plt.title("linear Regression ")
# plt.show()    ### we are not getting best fit line

##-----solve above problem by using polynomial Regression degree 4 based on accuracy we need to change degree val---------------
##apply polynomial transformation

# poly=PolynomialFeatures(degree=4,include_bias=True)
# X_train_poly=poly.fit_transform(X_train)
# X_test_poly=poly.transform(X_test)

##apply same linear regression model
# regression_poly=LinearRegression()
# regression_poly.fit(X_train_poly,Y_train)
# Y_pred_poly=regression_poly.predict(X_test_poly)
# score_poly=r2_score(Y_pred_poly,Y_test)
# print(score_poly)
# print(regression_poly.coef_)
# print(regression_poly.intercept_)
#
# plt.scatter(X_train,regression_poly.predict(X_train_poly),color='blue')
# plt.scatter(X_train,Y_train)
# plt.title("linear ditrubution vs polynomial distribution")
# plt.show()


###-------------prediction of new data---------------
# X_new=np.linspace(-3,3,200).reshape(200,1)
# X_new_poly=poly.transform(X_new)
# Y_new_pred=regression_poly.predict(X_new_poly)
# print(np.mean(Y_new_pred))




##-----------------pipeline concept for new data prediction-----------------------------

def poly_regression(degree):
    X_new_func=np.linspace(-3,3,200).reshape(200,1)
    poly_features= PolynomialFeatures(degree=4, include_bias=True)
    lin_regression=LinearRegression()
    poly_feature_regression=Pipeline([("poly_features",poly_features),('poly_regression',lin_regression)])
    poly_feature_regression.fit(X_train,Y_train)
    y_pred_poly=poly_feature_regression.predict(X_new_func)


    plt.scatter(X_train,Y_train)
    plt.plot(X_new_func,y_pred_poly,label=" degree "+str(degree),linewidth=3)
    plt.scatter(X_test,Y_test)
    plt.xlabel('x vals')
    plt.ylabel('y  vals')
    plt.title("how polynomial regression works")
    plt.show()


poly_regression(2)










