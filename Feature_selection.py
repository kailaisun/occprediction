import pandas as pd
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import os
import io
import seaborn as sns
import datetime
ExternalFiles_folder = r"D:\Tsinghua UNI\CFINS\CASE study\Original_data"
FileName = "room_7_result.csv"
path_File = os.path.join(ExternalFiles_folder,FileName)

DF_main= pd.read_csv(path_File,sep=";",index_col=0)
DATASET= pd.read_csv(path_File,usecols=['date','time','FCU_temp_feedback','FCU_control_mode','FCU_onoff_feedback','FCU_fan_feedback','occupant_num','room_temp1','room_RH1','room_temp2','room_RH2'])

FCU_temp = DATASET['FCU_temp_feedback']
FCU_control = DATASET['FCU_control_mode']
FCU_onoff = DATASET['FCU_onoff_feedback']
FCU_fan = DATASET['FCU_fan_feedback']
occupancy = DATASET['occupant_num']
room_temp1 = DATASET['room_temp1']
room_RH1 = DATASET['room_RH1']
room_temp2 = DATASET['room_temp2']
room_RH2 = DATASET['room_RH2']
# Calculate the Pearson correlation coefficient
pearson_corr1 = FCU_temp.corr(occupancy)
pearson_corr2 = FCU_control.corr(occupancy)
pearson_corr3 = FCU_onoff.corr(occupancy)
pearson_corr4 = FCU_fan.corr(occupancy)
pearson_corr5 = room_temp1.corr(occupancy)
pearson_corr6 = room_RH1.corr(occupancy)
pearson_corr7 = room_temp2.corr(occupancy)
pearson_corr8 = room_RH2.corr(occupancy)
print("Pearson Correlation Coefficient (FCU_temp):", round(pearson_corr1,3))
print("Pearson Correlation Coefficient (FCU_control):", round(pearson_corr2,3))
print("Pearson Correlation Coefficient (FCU_onoff):", round(pearson_corr3,3))
print("Pearson Correlation Coefficient (FCU_fan):", round(pearson_corr4,3))
print("Pearson Correlation Coefficient (room_temp1):", round(pearson_corr5,3))
print("Pearson Correlation Coefficient (room_RH1):", round(pearson_corr6,3))
print("Pearson Correlation Coefficient (room_temp2):", round(pearson_corr7,3))
print("Pearson Correlation Coefficient (room_RH2):", round(pearson_corr8,3))

x = occupancy
y = room_RH2
# Create a scatter plot with a Line of Best Fit
sns.regplot(x=x, y=y)

# Set labels and title
plt.xlabel('Occupant Number')
plt.ylabel('Relative humidity 2')
plt.title('Zone 7: scatter Plot with Line of Best Fit \n Correlation: {:.3f}'.format(pearson_corr8))
# Display the plot
plt.show()