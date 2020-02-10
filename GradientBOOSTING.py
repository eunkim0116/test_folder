import numpy as np
####
import pandas as pd
import matplotlib.pyplot as plt
import os
data = pd.read_csv("P:/breastCancer.csv")
data.drop(["id","Unnamed: 32"], axis=1,inplace=True)
data.diagnosis=[1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
#normalization
x_data = data.drop(["diagnosis"],axis=1)
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#train_test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15,random_state=41)

from sklearn.tree import DecisionTreeClassifier
df=DecisionTreeClassifier()
df.fit(x_train,y_train)
print(df.score(x_test,y_test))
#print(x)
