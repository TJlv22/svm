import tensorflow as tf
from tensorflow import keras
import pandas as pd 
import numpy as np
from keras import models
from keras.layers import LSTM,Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

data=pd.read_csv("usdjpyday2.csv")

#create data
datax=[]
datay=[]
p=20

for i in range(len(data)-p) :
    datax.append(data["lag1"][i:i+p])
    datay.append(data["direction"][i+p])
    
datax=np.array(datax)
datay=np.array(datay)

#split data
l=int(len(datax)*0.8)
xtrain=datax[:l]
xtest=datax[l:]
ytrain=datay[:l]
ytest=datay[l:]


#normalize data
scal=StandardScaler()
xtrain1=scal.fit_transform(xtrain)
xtest1=scal.fit_transform(xtest)


#change shape of data
xtrain1=np.reshape(xtrain1,(xtrain1.shape[0],1,xtrain1.shape[1]))
xtest1=np.reshape(xtest1,(xtest1.shape[0],1,xtest1.shape[1]))

ytrain=np.where(ytrain>0,1,0)
ytest=np.where(ytest>0,1,0)
ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)

#create model
model=models.Sequential()
model.add(LSTM(128,activation="tanh",input_shape=(1,p)))
#model.add(Dense(128,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="RMSprop")
#model.compile(loss="mean_squared_error", optimizer="adam")

#learn and plot
result=model.fit(xtrain1,ytrain,batch_size=200,epochs=300)
yp=model.predict(xtest1)


print(yp)
print(ytest)




