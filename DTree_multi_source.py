import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import io
import seaborn as sns
import datetime
ExternalFiles_folder = r"D:\Tsinghua UNI\CFINS\CASE study\occupanyprediction"
FileName = "room1result.csv"
path_File = os.path.join(ExternalFiles_folder,FileName)

DF_main= pd.read_csv(path_File,sep=";",index_col=0)
#DATASET= pd.read_csv(path_File,usecols=['date','time','FCU_temp_feedback','FCU_control_mode','FCU_onoff_feedback','FCU_fan_feedback','occupant_num','room_temp1','room_RH1','room_temp2','room_RH2'])
DATASET= pd.read_csv(path_File,usecols=['date','time','FCU_onoff_feedback','occupant_num'])
X1 = DATASET['occupant_num']
X2 = DATASET['FCU_onoff_feedback']

from sklearn.preprocessing import MinMaxScaler
#create numpy.ndarray
X1 = X1.values
X2 = X2.values
#train_data.values, test_data.values, all_data.values = train.astype('float32'), test.astype('float32'), all_days.astype('float32')
X1 = np.reshape(X1, (-1, 1))
X2 = np.reshape(X2, (-1, 1))
X1_scaler = MinMaxScaler(feature_range=(0, 1))
X2_scaler = MinMaxScaler(feature_range=(0, 1))
X1 = X1_scaler.fit_transform(X1)
X2 = X2_scaler.fit_transform(X1)

def convert2matrix(data_arr, look_back):
   X, Y =[], []
   for i in range(len(data_arr)-look_back):
       d=i+look_back
       X.append(data_arr[i:d,])
       Y.append(data_arr[d,])
   return np.array(X), np.array(Y)

# setup look_back window
look_back = 20 # each 20 minutes
#convert dataset into right shape in order to input into the DNN
occ, his_occ  = convert2matrix(X1, look_back)

X2 = pd.DataFrame(X2)
his_occ = pd.DataFrame(his_occ)

DATASET['FCU_on/off'] = X2
DATASET['Historical_occ'] = his_occ
DATASET['occupany'] = X1


DATASET = DATASET.drop('occupant_num', axis=1)
DATASET = DATASET.drop('FCU_onoff_feedback', axis=1)
print(DATASET)

train_data=DATASET.iloc[:5760]
test_data = DATASET.iloc[5760:7200]

indexes = [idx for idx in range(10080) if 540 <= idx%1440 <= 1140]
train_data=train_data.filter(items=indexes, axis=0)
test_data=test_data.filter(items=indexes, axis=0)
all_data = DATASET.filter(items=indexes, axis=0)

train_data['datetime'] = train_data[['date', 'time']].agg(' '.join, axis=1)
test_data['datetime'] = test_data[['date', 'time']].agg(' '.join, axis=1)
all_data['datetime'] = all_data[['date', 'time']].agg(' '.join, axis=1)

train_data['datetime']=train_data['datetime'].astype('datetime64[ns]')
test_data['datetime']=test_data['datetime'].astype('datetime64[ns]')
all_data['datetime']=all_data['datetime'].astype('datetime64[ns]')

ts_train, ts_test = train_data, test_data
ts = all_data
ts['datetime'] = pd.to_datetime(ts['datetime'])
ts_train['datetime'], ts_test['datetime'] = pd.to_datetime(ts_train['datetime']), pd.to_datetime(ts_test['datetime'])
ts_train.set_index('datetime', inplace=True)
ts_test.set_index('datetime', inplace=True)
ts.set_index('datetime', inplace=True)

NEWDATASET_train = ts_train.drop(['date','time'],axis=1)
NEWDATASET_test = ts_test.drop(['date','time'],axis=1)
NEWDATASET_all = ts.drop(['date','time'],axis=1)
train_data = NEWDATASET_train.fillna(0)
test_data =  NEWDATASET_test.fillna(0)
all_data = NEWDATASET_all.fillna(0)

trainX = train_data.loc[:,["occupany"]]
trainY = train_data.drop("occupany",axis=1)

trainX = trainX.values
#trainX = trainX.flatten()
trainY = trainY.values
#trainY = trainY.flatten()
testX = test_data.loc[:,["occupany"]]
testY = test_data.drop("occupany",axis=1)

testX = testX.values
#testX = testX.flatten()

testY = testY.values
#testY = testY.flatten()
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
trainY = np.reshape(trainY, (trainY.shape[0], 1, trainY.shape[1]))
testY = np.reshape(testY, (testY.shape[0], 1, testY.shape[1]))
#all_daysX = np.reshape(all_daysX, (all_daysX.shape[0], 1, all_daysX.shape[1]))

# Create a Decision Tree regressor object
dt_regressor = DecisionTreeRegressor()

trainX=trainX[:,:,0]
trainY=trainY[:,:,0]
# Fit the regressor with the training data
dt_regressor.fit(trainX, trainY)

testX=testX[:,:,0]
testY=testY[:,:,0]
# Use the fitted regressor to make predictions on the test data
predicted_Y = dt_regressor.predict(testX)
# Evaluate the accuracy of the predictions using mean squared error
mse = mean_squared_error(testY, predicted_Y)
# Print the mean squared error
print("Mean Squared Error:", mse)

r2 = r2_score(testY, predicted_Y)
print("R^2 Score:", r2)

#accuracy = accuracy_score(testY, predicted_Y)
#print('Accuracy:', accuracy)

def plotting(y_t, y_h):
    y_h = [int(idx + 0.5) for idx in y_h]
    # y_h = y_h.
    x = [idx for idx in range(len(y_t))]
    x_ticks = [i * 60 for i in range(11)]
    my_xticks = ['09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00']

    _, ax = plt.subplots()
    ax.plot(x, y_t, 'b', label="Ground Truth")
    ax.plot(x, y_h, 'g', label="Predictions")
    ax.set_xlabel('Working Time From 9am-6pm')
    ax.set_ylabel('Number of occupant in the room')
    ax.set_title('Comparison for zone 1')
    ax.legend(loc='upper right', shadow=True)
    plt.xticks(x_ticks, my_xticks)
    plt.show()
    # y_h=np.array(y_h)
    # accuracy = accuracy_score(y_t, y_h)
    # print('Accuracy:', accuracy)
plotting(testY, predicted_Y)
