import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle


print("Reading the dataset from csv file")
data = pd.read_csv("dataset/breast-cancer-dataset.csv")
print(data.shape)
print(data.describe())

print("<<<<< Checking for NULL values >>>>>")
print(data.isna().sum())

print("<<<<< As per EDA, the feature mean_area has some outliers")
print("removing top 10% outliers from the feature")
q = data["mean_area"].quantile(0.90)
data = data[data["mean_area"]<q]

        #splitting data into feature & outcome
X = data.drop("diagnosis",axis=1)
y = data["diagnosis"]


        #splitting data into train & test
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.25,random_state=355)

        #initializing the model for training
DT = DecisionTreeClassifier()
DT.fit(train_x,train_y)

print(DT.score(train_x,train_y))

print("<<<<< Saving model to the disk >>>>>")
model_file = "breast-cancer-prediction.pkl"
pickle.dump(DT,open(model_file,"wb"))



