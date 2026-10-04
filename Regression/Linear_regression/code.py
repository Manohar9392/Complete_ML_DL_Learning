import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score



df=pd.read_excel("dataset.xlsx")

#first check the relation
# plt.scatter(df['Weight'],df['Height'],color='red')
# plt.xlabel('Weights')
# plt.ylabel("heights")
# plt.title("heights vs weights relation")
# plt.show()


##check correlation
# print(df.corr())
# sns.pairplot(df)
# plt.show()


##----------Independent and dependent features--------------
X=df[['Weight']]        ##make independent features should be as dataframe  or 2d array
Y=df['Height']               ## series is fine  because it is one feature only


##-----------Train Text split----------------------
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42)  ##random_state is used to fix the train data and test data in same condition whenever excute



##---------standardization for Independent features-------------------
##to make all features  in same range    mean=0 standarddeviation =1
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)



##-------------Linear Regression Model------------------------
regression=LinearRegression(n_jobs=-1)
regression.fit(X_train,Y_train)

slope=regression.coef_
intercept=regression.intercept_

##----------------------ploting using training data
# plt.scatter(df['Weight'],df['Height'],color='red')
# plt.plot(X_train,regression.predict(X_train))
# plt.xlabel('Weights')
# plt.ylabel("heights")
# plt.title("Best fit line for training data")
# plt.show()

#------------------Prediction for test data
Y_pred=regression.predict(X_test)


##--------performance-metrics------------
mse=mean_squared_error(Y_test,Y_pred)
mae=mean_absolute_error(Y_test,Y_pred)
rmse=np.sqrt(mse)

score=r2_score(Y_test,Y_pred)


##-------------------------prediction for new data----------------------

print(regression.predict(scaler.transform([[72]])))   ##for new data standardiazation also done

