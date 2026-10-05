import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


df=pd.read_csv("economic_index.csv")
#print(df.head())

##------------------drop unneccessary columns--------------------------
df.drop(columns=["Unnamed: 0","year","month"],inplace=True)
# print(df.info())
# print(df.isnull().sum())

##----------------some visualizations----------------------
# sns.pairplot(df)
# plt.show()
# print(df.corr())
# sns.heatmap(df.corr())
# plt.show()
# plt.scatter(df['interest_rate'],df['unemployment_rate'],color='blue')
# plt.xlabel('interest rate')
# plt.ylabel('unemployement rate')
# plt.title("interest rate vs unemployement rate")
# plt.show()


##---------------independent and dependent features---------------------------
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]

##----------------train test split------------
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

# sns.regplot(df['interest_rate'],df['index_price'])
# sns.regplot(df['interest_rate'],df['unemployment_rate'])## linear regression plot
# sns.regplot(df['unemployment_rate'],df['index_price'])
# plt.show()

##------------standardization---------------------
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

##---------------model----------------------------
regression=LinearRegression(n_jobs=-1)
regression.fit(X_train,Y_train)

print(regression.coef_)
print(regression.intercept_)


##--------------cross validation------------------
from sklearn.model_selection import cross_val_score
score=cross_val_score(regression,X_train,Y_train,scoring="r2",cv=3)
print("Score",np.mean(score))


##--------------prediction------------------
Y_pred=regression.predict(X_test)

##------------performance_metrics------------------
mse=mean_squared_error(Y_test,Y_pred)
mae=mean_absolute_error(Y_test,Y_pred)
rmse=np.sqrt(mse)

score=r2_score(Y_test,Y_pred)

print(mse)
print(mae)
print(rmse)
print(score)



##---------------Assumptions to get conclusion----------------------
##scatter plot between y_pred and y_test if follows linear regression positive then model performing good
# plt.scatter(Y_pred,Y_test)
# residuals=Y_pred-Y_test
# sns.displot(residuals,kind='kde')
# plt.scatter(residuals,Y_pred)






#####--------------------------model by ols method ------------
# import statmodels.api as sm
# model=sm.OLS(X_train,Y_train).fit()
# print(model.summary)



