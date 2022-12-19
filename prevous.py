'''
from ast import Import
from base64 import standard_b64encode
from cgitb import text
import os
from random import random
from socket import SOMAXCONN
from tkinter import FALSE
from turtle import end_fill
from xml.etree.ElementPath import xpath_tokenizer
from gtts import gTTS
from playsound import playsound
import pyttsx3
import speech_recognition as sr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
'''



'''
def convert_status_to_description(x):
    if x['Index'] == 0:
        return 'Extremely Weak'
    elif x['Index'] == 1:
        return 'Weak'
    elif x['Index'] == 2:
        return 'Normal'
    elif x['Index'] == 3:
        return 'Overweight'
    elif x['Index']== 4:
        return 'Obesity'
    elif x['Index'] == 5:
        return 'Extreme Obesity'

def convert_gender_to_label(x):
    if x['Gender'] == 'Male':
        return 1
    elif x['Gender'] == 'Female':
        return 0

data = pd.read_csv('bmi.csv')
data_visual = pd.read_csv('bmi.csv')

data_visual['Status'] = data_visual.apply(convert_status_to_description,axis=1)
#print(data_visual.head())

data_visual['gender_lbl'] = data_visual.apply(convert_gender_to_label,axis=1)
#print(data_visual.head())


df = pd.read_csv('bmi.csv')


#print(df.sample(frac=0.1))# to display 10 percentage of data in the table
data=pd.get_dummies(df)
df = pd.DataFrame(data)
df.head()

std_sc = StandardScaler()
df.iloc[:,[0,1,3,4]] = std_sc.fit_transform(df.iloc[:,[0,1,3,4]])

#print(df.iloc[:,[0,1,3,4]])
#print(df.head()

X = df.iloc[:,[0,1,3,4]]
y = df.iloc[:,2]

#X = df.iloc[:,:-1]
#y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=0)

#print(type(X_train))
#print(X_train.head()) 



rfc = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
rfc.fit(X_train, y_train)

y_pred_rfc = rfc.predict(X_test)
#print(X_test)
rfc_cm = confusion_matrix(y_test, y_pred_rfc)
#print(rfc_cm)


rfc_acc = accuracy_score(y_test, y_pred_rfc)
#print(rfc_acc*100)


def health_test(gender, height, weight):

    individual_data_dict = {'Height':height, 'Weight':weight,'Gender':gender}   
    individual_data = pd.DataFrame(data = individual_data_dict, index=[2])
    lbl_enc = preprocessing.LabelEncoder()
    individual_data.iloc[:,2] = lbl_enc.transform(individual_data.iloc[:,2])
    #individual_data = one_hot_enc_for_gender.transform(individual_data).toarray()
    df = pd.DataFrame(individual_data)
    df.iloc[:,:] = std_sc.transform(df.iloc[:,:])
    y_pred = rfc.predict(individual_data)
    if y_pred == 0:
        return 'Extremely Weak'
    elif y_pred == 1:
        return 'Weak'
    elif y_pred == 2:
        return 'Normal'
    elif y_pred == 3:
        return 'Overweight'
    elif y_pred == 4:
        return 'Obesity'
    elif y_pred == 5:
        return 'Extreme Obesity'


sample_person = [155,78,'Female']
sample_result = health_test(*sample_person)
print(sample_result)

'''

'''
#Required libraries
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

#reading data from a file
df=pd.read_csv("bmi.csv")

#change gender column values

#considering Gender, Height and weight independent variables
x=df[["Height","Weight"]]

#dependent variable Bmi index 
y=df["Index"]

#split and train
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#using QuadraticDiscriminantAnalysis
regr=QuadraticDiscriminantAnalysis()
X_train = X_train.values
regr.fit(X_train, y_train)

#predict the Bmi Index
index=regr.predict([[165,78]])
print(index)
'''














'''
df = pd.read_csv('bmi.csv')
#df.shape
#print(df.describe())
#print(df.values)


#print(df.sample(frac=0.1))# to display 10 percentage of data in the table
df=pd.get_dummies(df)
#print(df)

X=df.iloc[:,[0,1,3,4]].values
Y=df.iloc[:,2].values

X_nu=df[["Height","Weight","Index"]]
X_nu.corr()


plt.hist(X_nu,bins=50)
plt.show()

plt.scatter(X_nu.Index,Y,color="g")
plt.grid()
plt.xlabel("Index")
plt.ylabel("Y(original result)")
plt.show()

plt.scatter(X_nu.Weight,Y,color="r")
plt.grid()
plt.xlabel("Weight")
plt.ylabel("Y(original result)")
plt.show()

plt.scatter(X_nu.Height,Y,color="teal")
plt.grid()
plt.xlabel("Height")
plt.ylabel("Y(original result)")
plt.show()



X_train=X[:400]
X_test=X[400:]

Y_train=Y[:400]
Y_test=Y[400:]

teacher=LinearRegression() 
learner=teacher.fit(X,Y) #student learning something about input & output data set to predict
learner_res = learner.predict(X_test) #after leaning student attending an exam


#conveting input as list 
xlist = list(X_train)
ylist = list(Y_train)

#conveting out as list
rlist = list(learner_res)


print(X_test)
print(end='\n')
print(rlist)
'''









'''
twilio
def sendMsg(phoneNumber,Name):
    account_sid = "AC283d53f8c2c3449426e40b2e4dce0887"
    auth_token = "f92b260f638fce2c56301543713c84bd"
    phone_number = "+91"+phoneNumber
    client = Client(account_sid,auth_token)
    
    sms = client.messages.create(
        from_ = "+17152008562",
        body = "Hi we are from Heath Checking System, Thank you for coming "+Name,
        to = phone_number)
    
'''






'''
Name = "kaasiprasanth"

body ="Hi "+Name+",You have checked (Height and weight ratio) in HEALTH CKECKING SYSTEM!. By your height and weight we're decided that you are in weak stage.To improve your health, you should eat below items in daily or weeky routine.\n "+"\n* Unprocessed foods.\n* Fruits and vegetables.\n* Non-caffeinated beverages.\n* Lean proteins.\n* Whole grains and complex carbs.\n* Nuts.\n* Water.\n* Vitamins and supplements.\nThank you!"               

print(body)
'''