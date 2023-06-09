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

from sklearn.preprocessing import MinMaxScaler

from occformer import TransformerModel,CustomDataset
from torch.utils.data import DataLoader, Dataset
import torch
import math
import torch.nn as nn



def convert2matrix(data_arr, look_back):
   X, Y =[], []
   for i in range(len(data_arr)-look_back):
       d=i+look_back
       X.append(data_arr[i:d,])
       Y.append(data_arr[d,])
   return np.array(X), np.array(Y)

def plotting(y_t, y_h,input_size,occupancy_index,test_scaler,ii):
    # y_h=y_h[:,np.newaxis]

    yy_t=np.empty((len(y_t), input_size))
    yy_h = np.empty((len(y_h), input_size))
    yy_t[:,occupancy_index]=y_t
    yy_h[:,occupancy_index]=y_h[:,0]
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
    ax.set_xlabel('Working Time From 9am-6pm')
    ax.set_ylabel('Number of occupant in the room')
    ax.set_title('Comparison for zone 1')
    ax.legend(loc='upper right', shadow=True)
    plt.xticks(x_ticks, my_xticks)
    # plt.show()

    # y_h= np.around(y_h,0).astype(int)
    # y_t=y_t.tolist()
    y_h=np.array(y_h)

    y_t=y_t.astype(int)
    accuracy = accuracy_score(y_t, y_h)
    print('Accuracy:', round(accuracy,3))
    mse = mean_squared_error(y_t, y_h)
    # Print the mean squared error
    print("Mean Squared Error:", round(mse, 3))

    result = pd.DataFrame({
        'True_value': y_t,
        'transformer': y_h,
    })
    # Save the DataFrame as a CSV file
    result.to_csv(str(input_size)+'features-T_Z'+str(room)+'_SameScale' + str(ii) + '.csv', index=True)











def dataprocess(path_File,input_size,room):
    DF_main = pd.read_csv(path_File, sep=";", index_col=0)
    # DATASET = pd.read_csv(path_File,
    #                       usecols=['date', 'time', 'FCU_temp_feedback', 'FCU_control_mode', 'FCU_onoff_feedback',
    #                                'FCU_fan_feedback', 'occupant_num', 'room_temp1', 'room_RH1', 'room_temp2',
    #                                'room_RH2'])
    if input_size==2:
        DATASET= pd.read_csv(path_File,usecols=['date','time','occupant_num', 'FCU_control_mode'])
    if input_size==9:
        DATASET = pd.read_csv(path_File,
                          usecols=['date', 'time', 'FCU_temp_feedback', 'FCU_control_mode', 'FCU_onoff_feedback',
                                   'FCU_fan_feedback', 'occupant_num', 'room_temp1', 'room_RH1', 'room_temp2',
                                   'room_RH2'])
    if input_size==1:
        DATASET = pd.read_csv(path_File,
                          usecols=['date', 'time','occupant_num'])

    if input_size==3:
        if room==1 or room==7:
            DATASET = pd.read_csv(path_File,
                              usecols=['date', 'time','occupant_num','FCU_onoff_feedback','room_temp1'])
        else:
            DATASET = pd.read_csv(path_File,
                              usecols=['date', 'time','occupant_num','FCU_onoff_feedback','FCU_control_mode'])

    if input_size==4:
        if room==1 or room==7:
            DATASET = pd.read_csv(path_File,
                              usecols=['date', 'time','occupant_num','FCU_onoff_feedback','room_temp1','room_temp2'])
        else:
            DATASET = pd.read_csv(path_File,
                              usecols=['date', 'time','occupant_num','FCU_onoff_feedback','FCU_control_mode','room_temp1'])

    if input_size==5:
        if room==1:
            DATASET = pd.read_csv(path_File,
                              usecols=['date', 'time','occupant_num','FCU_onoff_feedback','room_temp1','room_temp2','FCU_temp_feedback'])
        elif room==2:
            DATASET = pd.read_csv(path_File,
                              usecols=['date', 'time','occupant_num','FCU_onoff_feedback','FCU_control_mode','room_temp1','FCU_temp_feedback'])
        else:
            DATASET = pd.read_csv(path_File,
                              usecols=['date', 'time','occupant_num','FCU_onoff_feedback','FCU_control_mode','room_temp1','room_temp2'])

    # DATASET = pd.read_csv(path_File, usecols=['date', 'time','occupant_num', 'room_temp2','FCU_onoff_feedback','room_temp1','FCU_control_mode'])

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
    # test_scaler = MinMaxScaler(feature_range=(0, 1))
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
    # # setup look_back window
    look_back = 20 # each 20 minutes
    #convert dataset into right shape in order to input into the DNN
    trainX, trainY = convert2matrix(train, look_back)
    testX, testY = convert2matrix(test, look_back)
    all_daysX, all_daysY = convert2matrix(all_days, look_back)
    return trainX,trainY,testX,testY,occupancy_index

for room in [1,2,4,5,6,7]:
    for input_size in [1,2,3,4,5,9]:
        print('room'+str(room)+', input feature'+ str(input_size))

        ExternalFiles_folder = r"./Original_data"
        FileName = "room_"+str(room)+"_result.csv"
        path_File = os.path.join(ExternalFiles_folder,FileName)
        trainX1,trainY1,testX1,testY1,occupancy_index=dataprocess(path_File,input_size,room)

        ExternalFiles_folder = r"./occupanyprediction"
        FileName = "room"+str(room)+"result.csv"
        path_File = os.path.join(ExternalFiles_folder,FileName)
        trainX2,trainY2,testX2,testY2,occupancy_index=dataprocess(path_File,input_size,room)

        #trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        #testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        #all_daysX = np.reshape(all_daysX, (all_daysX.shape[0], 1, all_daysX.shape[1]))

        # Create a Decision Tree regressor object
        # dt_regressor = DecisionTreeRegressor()

        # trainX=trainX[:,:,0]
        # Fit the regressor with the training data
        # dt_regressor.fit(trainX, trainY)

        # testX=testX[:,:,0]
        # Use the fitted regressor to make predictions on the test data
        # predicted_Y = dt_regressor.predict(testX)
        # Evaluate the accuracy of the predictions using mean squared error


        trainX=np.concatenate((trainX1,trainX2),axis=0)
        trainY=np.concatenate((trainY1,trainY2),axis=0)
        testX=np.concatenate((testX1,testX2),axis=0)
        # testY=np.concatenate((testY1,testY2),axis=0)


        train_scaler = MinMaxScaler(feature_range=(0, 1))#LTSM is senstive to the scale of features
        label_scaler = MinMaxScaler(feature_range=(0, 1))#LTSM is senstive to the scale of features
        trainX_X=trainX.reshape((len(trainX)*20,-1))
        testX_X=testX.reshape((len(testX)*20,-1))


        train= train_scaler.fit_transform(trainX_X)
        test=train_scaler.transform(testX_X)

        trainY=label_scaler.fit_transform(trainY)
        testY1,testY2=label_scaler.transform(testY1),label_scaler.transform(testY2)

        trainX=train.reshape((len(trainX),20,-1))
        testX=test.reshape((len(testX),20,-1))

        testX1=testX[0:len(testX1),:,:]
        testX2=testX[len(testX1):,:,:]

        trainX=trainX.astype(np.float32)
        trainY=trainY.astype(np.float32)

        # Set up the model and training parameters

        output_size = 1
        d_model = 128
        nhead = 8
        num_layers = 6
        import torch.optim as optim
        device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
        # device = torch.device( "cuda")
        # model = t2TransformerTemperatureModel(input_size, output_size, d_model, nhead, num_layers,predict_seq=predict_seq)

        model=TransformerModel(input_dim=input_size, hidden_dim= 512, n_layers = num_layers, n_heads = nhead, dropout=0.3)
        model=model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(),lr=0.0003)


        dataset = CustomDataset(trainX, trainY)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        # Train the model
        num_epochs = 50
        min_loss_val = 10
        import copy
        for epoch in range(num_epochs):
            for inputs, targets in dataloader:
                # 清零梯度
                optimizer.zero_grad()
                # inputs=inputs.unsqueeze(dim=-1)

               # 前向传播
                inputs=inputs.to(device)
                targets=targets.to(device)
                outputs = model(inputs)

                # if epoch==7:
                #     print(1)
                # 计算损失
                targets= targets[:,occupancy_index]
                targets=targets.unsqueeze(-1)
                loss = criterion(outputs, targets)
                if min_loss_val>loss:
                    # print(1)
                    min_loss_val = loss
                    best_model = copy.deepcopy(model)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

            # 打印每个 epoch 的损失
            # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        model = best_model

        torch.save(model, str(input_size)+'feature-model'+str(room)+'.pth')

        for ii in range(2):
            if ii==0:
                testX=testX1.astype(np.float32)
                testY=testY1.astype(np.float32)
                # test_scaler=test_scaler1
            else:
                testX=testX2.astype(np.float32)
                testY=testY2.astype(np.float32)
                # test_scaler=test_scaler2


            with torch.no_grad():
                model.eval()
                model=model.to('cpu')
                testX=torch.from_numpy(testX)
                testY=testY[:,occupancy_index]
                # testX = testX.unsqueeze(dim=-1)
                predicted_Y=model(testX)



            # r2 = r2_score(testY, predicted_Y)
            # print("R^2 Score:", r2)
            plotting(testY, predicted_Y,input_size,occupancy_index,label_scaler,ii)
            #accuracy = accuracy_score(testY, predicted_Y)
            #print('Accuracy:', accuracy)

