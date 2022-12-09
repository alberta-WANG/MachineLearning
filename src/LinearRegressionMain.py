import os
import sys
sys.dont_write_bytecode = True
from Regression.LinearRegression import LinearRegression as lr

import data


myData = data.regression()
myData.makeData(dataType = 1)

dtrNum = int(len(myData.X)*0.9)

Xtr = myData.X[:dtrNum]
Ytr = myData.Y[:dtrNum]

Xte = myData.X[dtrNum:]
Yte = myData.Y[dtrNum:]

myModel = lr.linearRegression(Xtr,Ytr)
myModel.train()

print(f"model parameter:\nw={myModel.w},\nb={myModel.b}")
print(f"RMSE=${myModel.RMSE(Xte,Yte):.2f}")
print(f"coefficient of determination={myModel.R2(Xte,Yte):.2f}")

myModel.plotResult(X=Xtr,Y=Ytr,xLabel=myData.xLabel,yLabel=myData.yLabel,fName=f"../results/linearRegression_result_train_{myData.dataType}.pdf")