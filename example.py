import data
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
from keras import models, layers, callbacks 
from keras.callbacks import EarlyStopping
import keras 
from keras.optimizers import Adam, SGD
import seaborn as sns
import importlib
import matplotlib
import math
importlib.reload(data)

y,B=data.target(X,buoy,['swh','wspd'],['WVHT','WSPD'])

#import x

X_train, X_test, y_train, y_test=data.split_years(x,y,[2014,2015],[2016])    
Xtrain, Xtest, ytrain, ytest, input_scaler, output_scaler=data.scaled_dataset(StandardScaler(), 
                            StandardScaler(),X_train,X_test,y_train,y_test)

pca=PCA(.95) 
pca.fit(Xtrain)
Xtrain=pca.transform(Xtrain)
Xtest=pca.transform(Xtest)

outvars=['e_swh','e_wspd']
mergedT=pd.concat([pd.DataFrame(Xtrain),pd.DataFrame(ytrain,columns=outvars)],axis=1) 
mergedt=pd.concat([pd.DataFrame(Xtest),pd.DataFrame(ytest,columns=outvars)],axis=1)

n_in=3
aggT,aggt=data.series_to_supervised(mergedT,mergedt,['e_swh','e_wspd'], None,n_in=n_in,n_out=1)
aggTt, aggTv=train_test_split(aggT, test_size=.2,shuffle=False)
#split in x/y
trainX,trainY=aggTt.iloc[:,:-len(outvars)],aggTt.iloc[:,-len(outvars):]
valX,valY=aggTv.iloc[:,:-len(outvars)],aggTv.iloc[:,-len(outvars):]
testX,testY=aggt.iloc[:,:-len(outvars)],aggt.iloc[:,-len(outvars):]
#reshape
trainX=trainX.to_numpy().reshape((trainX.shape[0],n_in,mergedT.shape[1]))
valX=valX.to_numpy().reshape((valX.shape[0],n_in,mergedT.shape[1]))
testX=testX.to_numpy().reshape((testX.shape[0],n_in,mergedt.shape[1]))

#lstm
model=models.Sequential()
model.add(layers.LSTM(200,input_shape=(trainX.shape[1],trainX.shape[2]),dropout=0.2,kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01),return_sequences=True))
model.add(layers.LSTM(200,input_shape=(trainX.shape[1],trainX.shape[2]),dropout=0.2,kernel_regularizer=l2(0.01)))
model.add(layers.Dense(len(outvars)))
model.compile(loss='mse',optimizer=SGD(lr=0.001,momentum=0.95,nesterov=True))
monitor=EarlyStopping(monitor='val_loss',verbose=0,patience=20+n_in)
h = model.fit(trainX, trainY,callbacks=[monitor], epochs=300, batch_size=32, validation_data=(valX,valY), verbose=1, shuffle=False)
# plot history
plt.figure(figsize=(10,10))
plt.plot(h.history['loss'], label='train')
plt.plot(h.history['val_loss'], label='test')
plt.legend()
#predict
yhat=model.predict(testX)
inv_yhat=output_scaler.inverse_transform(yhat)
