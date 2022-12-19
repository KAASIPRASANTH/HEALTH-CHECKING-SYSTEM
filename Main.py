from random import random
from turtle import end_fill
from xml.etree.ElementPath import xpath_tokenizer
from gtts import gTTS
from playsound import playsound
import pyttsx3
import speech_recognition as sr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import csv
import numpy as np
from http import client
from twilio.rest import Client

import email
from email.message import EmailMessage
from operator import imod
import smtplib
from re import sub
import ssl

from tkinter import *
from tkinter import messagebox


def details():
    window = Tk()
    window.title('Health Checker')
    window.geometry('400x150')

    l1 = Label(window,text='Name : ',font=(14))
    l2 = Label(window,text='Mail Id : ',font=(14))

    l1.grid(row=0,column=0,padx=5,pady=5)
    l2.grid(row=1,column=0,padx=5,pady=5)

    Name = StringVar()
    Email = StringVar()

    t1 = Entry(window,textvariable=Name,font=(14))
    t2 = Entry(window,textvariable=Email,font=(14))

    t1.grid(row=0,column=1)
    t2.grid(row=1,column=1)
    def submit():
        messagebox.showwarning(title='Information',message='Thank You!')
        window.destroy()
    def cancel():
        status = messagebox.askyesno(title='Question',message='Do you want to close the window')
        if status == True:
            window.destroy()
        else:
            messagebox.showwarning(title='Warning Message',message='Please submit!')


    b1 = Button(window,command=submit,text='Submit',font=(14))
    b2 = Button(window,command=cancel,text='Cancel',font=(14))

    b1.grid(row=2,column=1,sticky=W)
    b2.grid(row=2,column=1,sticky=E)

    window.mainloop()
    v1 = str(Name.get())
    v2 = str(Email.get())
    list = [v1,v2]
    return list

def speech(text):
    engine = pyttsx3.init()
    engine.setProperty("rate",125)
    voices = engine.getProperty('voices')
    engine.setProperty("voice",voices[0].id)
    engine.say(text)
    engine.runAndWait()

def storeHeightAndWeight(Name,height,weight,index):
    field_names = ['Name','Height','Weight','Index']
    dict = {"Name": Name,"Height":height, "Weight":weight,"Index":index}
    with open('BMIDataBase.csv', 'a') as csv_file:
        print('kaasiprasanth')
        dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
        dict_object.writerow(dict)

def sendMail(Name,Email,body):
    email_sender = "heathchecker@gmail.com"
    email_password = "scqxvtlpqccpfoub"
    email_receiver = Email

    #subject = "Regarding Medical"
    #body = "Hi "+Name+"! Thank you for coming Health Checking System."

    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    #em['subject'] = subject

    em.set_content(body)
    
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com',465,context=context) as smtp:
        smtp.login(email_sender,email_password)
        smtp.sendmail(email_sender,email_receiver,em.as_string())
    

speech("Hi i am doctor of health checking system")
speech("can you Enter your name and Mail Id")
list = details()
Name = list[0]
Email = list[1]
speech("We are welcoming you "+Name+" for health checking hospital")

b = "Hi "+Name+"! Thank you for coming Health Checking System."

#sending mail
sendMail(Name,Email,b)


continueToCheck = True
while(continueToCheck):
    
    typeOfChecking = ["Diabetes checking","Height and Weight ratio check and we will provide what are all the foods you should take further","health checking by symptoms",]
    i=1
    for val in typeOfChecking:
        print(str(i)+") "+val,end='\n')
        i=i+1
    
    speech("In health checking system we are processing 3 types of checking")
    speech("One is "+typeOfChecking[0])
    speech("next is "+typeOfChecking[1])
    speech("last we have "+typeOfChecking[2])
    speech("What type of checking you need opition 1 or option 2 or potion 3 ")
    speech("Tell me "+Name)
    
    option  = input()

    if option == "1":
        print("diet")




    elif option == "2":
        body = ""
        speech("Thank you for selecting option 2")
        speech("Enter your Height ")
        height = int(input())
        speech("Enter your Weight ")
        weight = int(input())
        speech("We are processing your data")
        speech("Kindly wait few seconds ")


        df=pd.read_csv("bmi.csv")
        x=df[["Height","Weight"]]
        y=df["Index"]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


        #Algorithm for BMI
        regr = QuadraticDiscriminantAnalysis()
        X_train = X_train.values
        regr.fit(X_train, y_train)

        #predict the Bmi Index
        index=regr.predict([[height,weight]])

        # BMI Index
        result = ["Extremely weak","Weak","Normal","Overweight","Obesity","Extreme obesity"]
        speech("Your results are ready "+Name)
        res = result[index[0]]
        speech("You are in "+res+" Stage")
        
        #print('BMI index ',index[0],res)
        storeHeightAndWeight(Name,height,weight,index[0]+1)
        


        if index[0]==0 or index[0]==1:
            body = "Hi "+Name+",You have checked (Height and weight ratio) in HEALTH CKECKING SYSTEM!. By your height and weight we're decided that you are in weak stage.To improve your health, you should eat below items in daily or weeky routine.\n "+"\n* Unprocessed foods.\n* Fruits and vegetables.\n* Non-caffeinated beverages.\n* Lean proteins.\n* Whole grains and complex carbs.\n* Nuts.\n* Water.\n* Vitamins and supplements.\nThank you!"
            sendMail(Name,Email,body)
        elif index[0] == 2:
            body = "Hi "+Name+",You have checked (Height and weight ratio) in HEALTH CKECKING SYSTEM!.  By your height and weight we're decided that you are in Normal stage.\nThank You!"
            sendMail(Name,Email,body)
        elif index[0] == 3 or index[0] == 4 or index[0] == 5:
            body = "Hi "+Name+",You have checked (Height and weight ratio) in HEALTH CKECKING SYSTEM!.  By your height and weight we're decided that you are in weight.\n*Lean protein sources like chicken, turkey and grass-fed lean beef help keep you full, decrease cravings and stabilize blood sugar, says Feit.\n*Eggs.\n*Vegetables.\n*Avocados.\n*Apples.\n*Berries.\n*Nuts and Seeds.\n*Salmon.\nThank you!."
            sendMail(Name,Email,body)
       


    elif option == "3":
        speech("Thank you for selecting option 3")
        train = pd.read_csv("Training.csv")
        test = pd.read_csv("Testing.csv")
        train = train.drop(["Unnamed: 133"],axis=1)

        multiple_diseases_dataset = train

        #seaparating data set
        X = multiple_diseases_dataset.drop(["prognosis"],axis=1)
        Y = multiple_diseases_dataset[["prognosis"]]

        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

        #Algorithm
        dtc = DecisionTreeClassifier(random_state=42)
        model_dtc = dtc.fit(X_train,Y_train)
        speech("There are one thirty symptoms are there")
        speech("You should select at least two")


        print('Select Atleat two :\n')
        #input stage
        list_of_diseases = ('itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 
        'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze')
        input_data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        

        speech("Please select the symptoms")
        for i in range(1,133):
            print(i,list_of_diseases[i-1])


        for num_str in input().split():
            num_int = int(num_str)
            input_data[num_int-1] = 1
        

        input_data = tuple(input_data)
        input_data_as_numpy_array = np.asarray(input_data)
        #reshapw the array as we are predicting the one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        speech("We are processing your data")
        speech("Kindly wait few seconds ")

        predict = model_dtc.predict(input_data_reshaped)
        result = str(predict[0])
        speech("After analysing your symtoms")
        speech("We have decided you are having ")
        speech(result)
        body = "Hi "+Name+", you have checked your health by symtoms in HEALTH CKECKING SYSTEM!. Finally! we decided that you're having "+result+".\nThank you!"
        sendMail(Name,Email,body)


    else:
        speech("There is no option like this "+Name+"")



    speech("Are you want to continue "+Name)
    yesOrNo = input("Enter yes or no? :\n")

    if(yesOrNo == "no"):
        continueToCheck = False
    else:
        speech("Okey")
        speech("Let's continue")

speech("Thank you for comming health checking hospital")
