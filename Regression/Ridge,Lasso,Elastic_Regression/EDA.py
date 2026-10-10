import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv",header=1)  #take second row as attributes
#print(df.info())

#------- data cleaning------------
# print(df[df.isnull().any(axis=1)])  ##checking for null vals

df.loc[:122,"Region"]=0
df.loc[123:,"Region"]=1          #there are two categories thats why region
df["Region"]=df["Region"].astype(int)
#print(df.info())


df=df.drop([122,123]).reset_index(drop=True)
#print(df[df.isnull().any(axis=1)])
df=df.drop(165).reset_index(drop=True)              ### removing another row which contains nan value
#print(df.info())

####------------fix spaces in column names--------------
df.columns=df.columns.str.strip()


##---------------change datatypes of columns------------------
df[['month','year','day','Temperature','RH','Ws']]=df[['month','year','day','Temperature','RH','Ws']].astype(int)
object_cols=[features for features in df.columns if df[features].dtypes=='O']
for i in object_cols:
    if i!='Classes':
        df[i]=df[i].astype(float)
print(df.info())
df.to_csv("Algerian_forest_fires_dataset_Cleaned.csv",index=False)