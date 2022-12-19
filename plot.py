import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
stalking = pd.read_csv('HeightAndWeight.csv')
user = "kaasiprasanth"
user_details = stalking[stalking['Name']==user]


'''
print('Name: {}'.format(user_details['Name']))
print('Height: {}'.format(user_details['Height']))
print('Weight: {}'.format(user_details['Weight']))
'''


#print(type(user_details)) #data frame type

lh = user_details['Weight'].to_list()
lw = user_details['Height'].to_list()

print(lh)
print(lw)

xpoints = np.array(lh)
ypoints = np.array(lw)

plt.plot(xpoints, ypoints)
plt.show()

