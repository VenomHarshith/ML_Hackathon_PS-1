# ML_Hackathon_PS-1
Brest Cancer detection

# This code is for importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# For loading the dataset
data= pd.read_csv('data.csv')

# For changing catergorical data to numerical data using pandas
data['diagnosis']= data['diagnosis'].map({'M':1 ,'B':0})

# For checking data format
data.head()

# Seperating the features and traget variable and removing an extra column containing NaN(Not a Number) values
X=data.drop('diagnosis',axis=1)
x=X.drop('Unnamed: 32',axis=1)
y=data['diagnosis']

# For splitting the data train the model and to test the model
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

# For scaling the values using z_score normalization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# For creating logestic regression classifier
logistic_detection = LogisticRegression(random_state=42)

# For fitting the model and predicting whether the cancer is benign or malignant. Finding the accuracy and printing the report
logistic_detection.fit(x_train,y_train)
prediction=logistic_detection.predict(x_test)
acc=accuracy_score(y_test,prediction)
report= classification_report(y_test,prediction)
print("Accuracy:", acc)
print(report)

