import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#loading the datasets to a pandas dataFrame
diabetes_dataset = pd.read_csv("diabetes.csv")

#separating the data and labels
X = diabetes_dataset.drop(columns='Outcome',axis=1)
Y = diabetes_dataset['Outcome']

#Data Standardization
scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_dataset['Outcome']

#Train Test Split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,stratify = Y,random_state = 2)


#Training an model
classifier = svm.SVC(kernel = 'linear')  

#Training the support vector Machine Classifier
classifier.fit(X_train,Y_train)

#Accuracy Score for training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

#Accuracy score data Test data
X_text_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_text_prediction,Y_test)


input_data = (45,87,23,67,90,32,21,56)
#changing input data to numpy array

input_data_as_numpy_array = np.asarray(input_data)
#reshapw the array as we are predicting the one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")