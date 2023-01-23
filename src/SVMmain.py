import sys
sys.dont_write_bytecode = True
from Classification.SVM import SVM as svm
import data
import numpy as np

myData = data.classification(negLabel=-1,posLabel=1)
myData.makeData(dataType=3)

dNum = int(len(myData.X)*0.9)
Xtr = myData.X[:dNum]
Ytr = myData.Y[:dNum]
Xte = myData.X[dNum:]
Yte = myData.Y[dNum:]

xMean = np.mean(Xtr,axis=0)
xStd = np.std(Xtr,axis=0)
Xtr = (Xtr-xMean)/xStd
Xte = (Xte-xMean)/xStd

myModel = svm.SVM(Xtr,Ytr)
myModel.train()

print(f"parameter:\nw = {myModel.w}\nb = {myModel.b}")
print(f"accuracy={myModel.accuracy(Xte,Yte):.2f}")

myModel.plotModel2D(X=Xtr,
                    Y=Ytr,
                    spptInds=myModel.spptInds,
                    xLabel=myData.xLabel,
                    yLabel=myData.yLabel,
                    title="result",
                    fName=f"../results/SVM_result_{myData.dataType}.pdf",
                    isLinePlot=True)