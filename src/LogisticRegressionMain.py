import numpy as np
import sys
sys.dont_write_bytecode = True
from Regression.LogisticRegression import LogisticRegression as lr
import data

# 1. preparing data
myData = data.classification(negLabel=0,posLabel=1)
myData.makeData(dataType=1)

# 2. preparing training data and test data
dtrNum = int(len(myData.X)*0.9) # number of training data
# preparing training data
Xtr = myData.X[:dtrNum]
Ytr = myData.Y[:dtrNum]
# preparing test data
Xte = myData.X[dtrNum:]
Yte = myData.Y[dtrNum:]

# 3. standardizing data
xMean = np.mean(Xtr,axis=0)
xStd = np.std(Xtr,axis=0)
Xtr = (Xtr-xMean)/xStd
Xte = (Xte-xMean)/xStd

# 4. training model
myModel = lr.logisticRegression(Xtr,Ytr)
trLoss = []
teLoss = []
for ite in range(1001):
    trLoss.append(myModel.CE(Xtr,Ytr))
    teLoss.append(myModel.CE(Xte,Yte))
    if ite % 100 == 0:
        print(f"step:{ite}")
        print(f"model parameter:\nw={myModel.w}\nb={myModel.b}")
        print(f"loss ={myModel.CE(Xte,Yte):.2f}")
        print(f"accuracy={myModel.accuracy(Xte,Yte):.2f}")
        print("---------------------")

        # update model parameter
    myModel.update(alpha=1)

#-------------------
# 5. ploting real value and predict
if Xtr.shape[1] == 1:
    myModel.plotModel1D(X=Xtr,Y=Ytr,xLabel=myData.xLabel,yLabel=myData.yLabel,fName=f"../results/logistic_result_train_{myData.dataType}.pdf")
elif Xtr.shape[1] == 2:
    myModel.plotModel2D(X=Xtr,Y=Ytr,xLabel=myData.xLabel,yLabel=myData.yLabel,fName=f"../results/logistic_result_train_{myData.dataType}.pdf")
#-------------------

#-------------------
# 6. ploting train loss and estimate loss
myModel.plotEval(trLoss,teLoss,fName=f"../results/logistic_CE_{myData.dataType}.pdf")
#-------------------
