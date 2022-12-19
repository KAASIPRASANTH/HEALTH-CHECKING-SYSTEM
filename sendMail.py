import email
from email.message import EmailMessage
from multiprocessing import context
from operator import imod
import os
import smtplib
from re import sub
import smtplib
import ssl
email_sender = "heathchecker@gmail.com"
email_password = "scqxvtlpqccpfoub"
email_receiver = "kaarthikeyan.arcse2020@sece.ac.in"


subject = "Regarding Medical"
body = "Hi kaarthikeyan! Thank you for coming Health Checking System."


em = EmailMessage()
em['From'] = email_sender
em['To'] = email_receiver
em['subject'] = subject

em.set_content(body)

context = ssl.create_default_context()

with smtplib.SMTP_SSL('smtp.gmail.com',465,context=context) as smtp:
    smtp.login(email_sender,email_password)
    smtp.sendmail(email_sender,email_receiver,em.as_string())
    
print("Mail sent")