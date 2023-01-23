import numpy as np
import pandas as pd
import sys
sys.dont_write_bytecode = True
from Classification.NaiveBayes import naiveBayes as naiveBayes
import data

myData = data.sentimentLabelling()
myData.makeData(dataType=1)

dNum = int(len(myData.X)*0.9)
Xtr = myData.X[:dNum]
Ytr = myData.Y[:dNum]
Xte = myData.X[dNum:]
Yte = myData.Y[dNum:]

priors = np.array([[0.5,0.5]])
myModel = naiveBayes.NaiveBayes(Xtr,Ytr,priors)
myModel.train()

print(f"training data accuracy = {myModel.accuracy(Xtr,Ytr):.2f}")
print(f"test data accuracy = {myModel.accuracy(Xte,Yte):.2f}")