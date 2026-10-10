
"""Cross_validation  is basically used to hyperparameter tuning the model"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV,RidgeCV,ElasticNetCV
from sklearn.metrics import mean_absolute_error,r2_score




df=pd.read_csv("Algerian_forest_fires_dataset_Cleaned.csv")
# print(df.info())

##-------drop unneccessary columns--------------
df.drop(columns=['year','month','day'],inplace=True)

####-------Encoding Classes----------------
df['Classes']=df['Classes'].str.strip().map({'fire':1,"not fire":0}).astype(int)
# print(df.info())

##--------dividing independent and dependent features-------------------
X=df.drop(columns=['FWI'])
Y=df['FWI']


###---train test split----------
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

####-----------feature selection based on correlation-------------------

# plt.figure(figsize=(12,10))
# corr=X_train.corr()
# sns.heatmap(corr,annot=True)
# plt.title("checking for Correlality between features")
# plt.show()

####-------check for multicollinearty-------------------
def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                col_name=corr_matrix.columns[i]
                col_corr.add(col_name)
    return col_corr
corr_features=correlation(X_train,0.85)    ##this threshold will given by domain expertize

## drop features where correlation is more than 85 %
X_test.drop(columns=['BUI','DC'],inplace=True)
X_train.drop(columns=['BUI','DC'],inplace=True)


##------------------ standardization------------
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


####---------LassoCv for  hypertuning the model--------
lassocv=LassoCV(cv=5)   ## based on requirement we need to select parameters by default cv value also 5
lassocv.fit(X_train_scaled,Y_train)
Y_pred=lassocv.predict(X_test_scaled)
print(lassocv.alpha_)
print(lassocv.selection)
print(lassocv.mse_path_)
print("Accuracy_after_cv: ",r2_score(Y_pred,Y_test))



##--------------RidgeCv is also hypertuning-----------
ridgecv=RidgeCV(cv=5)           ##by default cv is 1 only takes 1 record at a time
ridgecv.fit(X_train_scaled,Y_train)
Y_pred=lassocv.predict(X_test_scaled)

print(ridgecv.alpha_)     ### this is key to hypertune selected best alpha val
print(ridgecv._estimator_type)
print("Accuracy_after_cv: ",r2_score(Y_pred,Y_test))



#--------------ElastinetCv is also hypertuning-----------
elasticnetcv=ElasticNetCV(cv=5)           ##by default cv is 1 only takes 1 record at a time
#here two vals will be the l1ration and l2ratio for ridge and lasso contribution
elasticnetcv.fit(X_train_scaled,Y_train)
Y_pred=elasticnetcv.predict(X_test_scaled)

print(elasticnetcv.alpha_)     ### this is key to hypertune selected best alpha val for better model this to be submitted in normal lasso or ridge
print(elasticnetcv.alphas_)
print("Accuracy_after_cv: ",r2_score(Y_pred,Y_test))