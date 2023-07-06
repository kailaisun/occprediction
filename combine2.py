import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io


LR_Fri = pd.read_csv('LR1.csv')
# print(LR_Fri)
LR_Fri = LR_Fri.iloc[0:579,:]

y_true = LR_Fri['True_value']
LR = LR_Fri["Linear_regression"].values

RF_Fri = pd.read_csv('RF1.csv')
RF_Fri = RF_Fri.iloc[0:579,:]
RF = RF_Fri["Random_Forest"].values


def plotting(y_t, y1, y2):
    x = [idx for idx in range(len(y_t))]
    x_ticks = [i * 60 for i in range(11)]
    my_xticks = ['09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00']

    _, ax = plt.subplots()
    ax.plot(x, y_t, 'b', label="Ground Truth")
    ax.plot(x, y1, 'g', label="linear regression")
    ax.plot(x, y2, 'y', label="random forest")
    # ax.plot(x, y3, 'r', label="XGBoost")
    # ax.plot(x, y4, 'm', label="Decision Tree")

    ax.set_xlabel('Working Time From 9am-7pm')
    ax.set_ylabel('Number of occupant in the room')
    ax.set_title('Comparison for zone 1')
    ax.legend(loc='upper right', shadow=True)
    plt.xticks(x_ticks, my_xticks)
    plt.show()

plotting(y_true, LR, RF)



