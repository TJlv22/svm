from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data=pd.read_csv("usdjpyday4.csv")
rsidata=data[["rsi"]]
directiondata=data["direction"]
returndata=data["return"]

rsidata=rsidata.values
directiondata=directiondata.values
returndata=returndata.values
#split data to traindata and testdata 
l=int(len(rsidata)*0.8)
xtrain=rsidata[:l]
xtest=rsidata[l:]
ytrain=directiondata[:l]
ytest=directiondata[l:]
returntest=returndata[l:]
#fit and predict
model=svm.SVC(C=1,gamma="scale")
model.fit(xtrain,ytrain)
pre=model.predict(xtest)

#plot result
#one long position
a=returntest.cumsum()
#algo position
b=pre*returntest
b=b.cumsum()

plt.figure(figsize=(10,6))
plt.plot(b,label="algotrade")
plt.plot(a,label="realtrada")
plt.legend()
c=pre==ytest
print(np.count_nonzero(c)/len(c))