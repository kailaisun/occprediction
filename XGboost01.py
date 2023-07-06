import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import io
import seaborn as sns
import datetime
from sklearn.preprocessing import MinMaxScaler

def convert2matrix(data_arr, look_back):
   X, Y =[], []
   for i in range(len(data_arr)-look_back):
       d=i+look_back
       X.append(data_arr[i:d,])
       Y.append(data_arr[d,])
   return np.array(X), np.array(Y)

input_size=4
def dataprocess(path_File):
    # DF_main = pd.read_csv(path_File, sep=";", index_col=0)
    # DATASET = pd.read_csv(path_File,
    #                       usecols=['date', 'time', 'FCU_temp_feedback','FCU_onoff_feedback',
    #                                'occupant_num', 'room_temp1', 'room_temp2'])
    # DATASET = pd.read_csv(path_File,
    #                       usecols=['date', 'time','FCU_temp_feedback','FCU_control_mode','FCU_onoff_feedback',
    #                                'occupant_num', 'room_temp1'])
    # DATASET = pd.read_csv(path_File,
    #                       usecols=['date', 'time', 'FCU_control_mode','FCU_onoff_feedback',
    #                                'occupant_num', 'room_temp1', 'room_temp2'])
    # DATASET = pd.read_csv(path_File,
    #                       usecols=['date', 'time', 'FCU_temp_feedback', 'FCU_control_mode', 'FCU_onoff_feedback',
    #                                'FCU_fan_feedback', 'occupant_num', 'room_temp1', 'room_RH1', 'room_temp2',
    #                                'room_RH2'])
    # DATASET= pd.read_csv(path_File,usecols=['date','time','occupant_num'])
    # z1,z7
    DATASET = pd.read_csv(path_File,
                          usecols=['date','time','FCU_onoff_feedback','occupant_num','room_temp1', 'room_temp2'])
    #z2,z4,z5,z6
    # DATASET = pd.read_csv(path_File,
    #                       usecols=['date','time','FCU_onoff_feedback','FCU_control_mode','occupant_num','room_temp1'])


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


    train, test, all_days = train_data.values, test_data.values, all_data.values
    #train_data.values, test_data.values, all_data.values = train.astype('float32'), test.astype('float32'), all_days.astype('float32')
    # train, test, all_days = np.reshape(train, (-1, 1)), np.reshape(test, (-1, 1)), np.reshape(all_days, (-1, 1))  #LTSM requires more input features compared to RNN or DNN
    # train_scaler = MinMaxScaler(feature_range=(0, 1))#LTSM is senstive to the scale of features
    # test_scaler = train_scaler
    # alldata_scaler = MinMaxScaler(feature_range=(0, 1))
    # train, test, all_days = train_scaler.fit_transform(train), test_scaler.fit_transform(test), alldata_scaler.fit_transform(all_days)

    ct=train_data.columns.values
    for ii in range(len(ct)):
        if ct[ii]=='occupant_num':
            break
    occupancy_index=ii
    print(ct,occupancy_index)
    # print(train_scaler.data_max_)
    # print(test_scaler.data_max_)
    # setup look_back window
    look_back = 20 # each 20 minutes
    #convert dataset into right shape in order to input into the DNN
    trainX, trainY = convert2matrix(train, look_back)
    testX, testY = convert2matrix(test, look_back)
    all_daysX, all_daysY = convert2matrix(all_days, look_back)
    return trainX,trainY,testX,testY,occupancy_index

# # setup look_back window
# look_back = 20 # each 20 minutes
# #convert dataset into right shape in order to input into the DNN
# trainX, trainY = convert2matrix(train, look_back)
# testX, testY = convert2matrix(test, look_back)
# all_daysX, all_daysY = convert2matrix(all_days, look_back)

#trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#all_daysX = np.reshape(all_daysX, (all_daysX.shape[0], 1, all_daysX.shape[1]))

ExternalFiles_folder = r"\"
FileName = "\"
path_File = os.path.join(ExternalFiles_folder,FileName)
trainX1,trainY1,testX1,testY1,occupancy_index=dataprocess(path_File)

ExternalFiles_folder = r"\"
FileName = "\"
path_File = os.path.join(ExternalFiles_folder,FileName)
trainX2,trainY2,testX2,testY2,occupancy_index=dataprocess(path_File)


trainX=np.concatenate((trainX1,trainX2),axis=0)
trainY=np.concatenate((trainY1,trainY2),axis=0)

testX=np.concatenate((testX1,testX2),axis=0)
testY=np.concatenate((testY1,testY2),axis=0)


train_scaler = MinMaxScaler(feature_range=(0, 1))#LTSM is senstive to the scale of features
label_scaler = MinMaxScaler(feature_range=(0, 1))#LTSM is senstive to the scale of features
trainX_X=trainX.reshape((len(trainX)*20,-1))
testX_X=testX.reshape((len(testX)*20,-1))

#data transform
train=train_scaler.fit_transform(trainX_X)
test=train_scaler.transform(testX_X)
#  label transform
trainY=label_scaler.fit_transform(trainY)
testY1,testY2=label_scaler.transform(testY1),label_scaler.transform(testY2)
testY=label_scaler.transform(testY)
trainX=train.reshape((len(trainX),20,-1))
testX=test.reshape((len(testX),20,-1))


trainY=trainY[:,occupancy_index]
testY=testY[:,occupancy_index]
# Create a Decision Tree regressor object
trainX=trainX.reshape((len(trainX),-1))
testX=testX.reshape((len(testX),-1))


# Convert the data into DMatrix format for XGBoost
# dtrain = xgb.DMatrix(trainX, label=trainY)
# dtest = xgb.DMatrix(testX, label=testY)
#
# # Define the parameters for the XGBoost model
# params = {
#     'objective': 'reg:squarederror',
#     'max_depth': 5,
#     'learning_rate': 0.1,
#     'n_estimators': 100,
#     'seed': 42
# }
#
# # Train the XGBoost model
# XGboost = xgb.train(params, dtrain)
#
# # Make predictions on the test data
# predicted_Y = XGboost.predict(dtest)


xgb_regressor = XGBRegressor()
# trainX=trainX[:,:,0]
# Fit the regressor with the training data
xgb_regressor.fit(trainX, trainY)

# testX=testX[:,:,0]
# Use the fitted regressor to make predictions on the test data
predicted_Y = xgb_regressor.predict(testX)

# Evaluate the accuracy of the predictions using mean squared error
# mse = mean_squared_error(testY, predicted_Y)
# # Print the mean squared error
# print("Mean Squared Error:", mse)

# r2 = r2_score(testY, predicted_Y)
# print("R^2 Score:", r2)

# accuracy = accuracy_score(testY, predicted_Y)
# print('Accuracy:', accuracy)

def plotting(y_t, y_h,input_size,occupancy_index,test_scaler,ii):
    # y_h=y_h[:,np.newaxis]
    #
    yy_t=np.empty((len(y_t), input_size))
    yy_h = np.empty((len(y_h), input_size))
    yy_t[:,occupancy_index]=y_t
    yy_h[:,occupancy_index]=y_h
    y_t=test_scaler.inverse_transform(yy_t)
    y_h = test_scaler.inverse_transform(yy_h)
    y_t=y_t[:,occupancy_index]
    y_h=y_h[:,occupancy_index]
    y_h=[int(idx+0.5) for idx in y_h]
    x = [idx for idx in range(len(y_t))]
    x_ticks = [i * 60 for i in range(11)]
    my_xticks = ['09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00']

    _, ax = plt.subplots()
    ax.plot(x, y_t, 'b', label="Ground Truth")
    ax.plot(x, y_h, 'g', label="Predictions")
    ax.set_xlabel('Working Time From 9am-7pm')
    ax.set_ylabel('Number of occupant in the room')
    ax.set_title('Comparison for zone 1')
    ax.legend(loc='upper right', shadow=True)
    plt.xticks(x_ticks, my_xticks)
    plt.show()

    # y_h= np.around(y_h,0).astype(int)
    # y_t=y_t.tolist()
    y_h=np.array(y_h)

    y_t=y_t.astype(int)
    accuracy = accuracy_score(y_t, y_h)
    print('Accuracy:', round(accuracy,3))

    mse = mean_squared_error(y_t, y_h)
    # Print the mean squared error
    print("Mean Squared Error:", round(mse, 3))

    result1 = pd.DataFrame({
        'True_value': y_t,
        'xgboost': y_h,
    })
    # Save the DataFrame as a CSV file
    result1.to_csv('XG_F4_Z7_SameScale'+str(ii)+'.csv', index=True)


for ii in range(2):
    if ii==0:
        # testX=testX1.astype(np.float32)
        # testY=testY1.astype(np.float32)
        # test_scaler=train_scaler
        testYY=testY[0:int(0.5*len(testY))]
        predicted_YY=predicted_Y[0:int(0.5*len(testY))]
    else:
        # testX=testX2.astype(np.float32)
        # testY=testY2.astype(np.float32)
        # test_scaler=test_scaler2
        testYY=testY[int(0.5*len(testY)):-1]
        predicted_YY = predicted_Y[int(0.5*len(testY)):-1]


    # r2 = r2_score(testYY, LR_predicted_YY)
    # print("R^2 Score:", r2)

    # LR_predicted_YY = np.round(LR_predicted_YY)
    plotting(testYY, predicted_YY, input_size, occupancy_index, label_scaler, ii)
    #accuracy = accuracy_score(testY, predicted_Y)
    #print('Accuracy:', accuracy)