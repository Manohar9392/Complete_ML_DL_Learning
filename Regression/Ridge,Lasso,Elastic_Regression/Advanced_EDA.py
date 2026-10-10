import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('Algerian_forest_fires_dataset_Cleaned.csv')

df_copy=df.copy()
print(df['Classes'].unique())
df_copy.drop(columns=['year','day','month'],inplace=True)    ##removing unneccessary_columns


##---------Encoding-----------
df_copy['Classes']=df_copy['Classes'].str.strip()     ##removing spaces
#print(df_copy['Classes'].unique())
df_copy['Classes']=df_copy["Classes"].map({'not fire':0,"fire":1})
df_copy['Classes']=df_copy['Classes'].astype(int)
# print(df_copy.info())

###--visualizations--
# plt.style.use('seaborn-v0_8')
# df_copy.hist(bins=50,figsize=(20,15))
#plt.show()



## percentage for pie chart
# pie_val=df_copy['Classes'].value_counts().reset_index()
#
# plt.pie(pie_val['count'],labels=pie_val['Classes'],explode=[0,0.1],shadow=True,autopct='%1.1f%%')
# plt.title("Fire vs not Fire")
# plt.show()


####-------------correlation-------
# print(df_copy.corr())
# sns.heatmap(df_copy.corr())
# plt.show()

#########---boxplot for outliers-----------
# sns.boxplot(df_copy['RH'])
# plt.show()


##---monthly fire analysis-----------
# df['Classes']=df['Classes'].str.strip()
# sns.countplot(x='month',hue='Classes',data=df)
# plt.xlabel("months")
# plt.ylabel("number of fires")
# plt.show()





