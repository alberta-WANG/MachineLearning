import numpy as np
import matplotlib.pylab as plt

class linearRegression():
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.dNum = X.shape[0]
        self.xDim = X.shape[1]

    def train(self):
        Z = np.concatenate([self.X,np.ones([self.dNum,1])],axis=1)
        ZZ = 1/self.dNum * np.matmul(Z.T,Z)
        ZY = 1/self.dNum * np.matmul(Z.T,self.Y)
        v = np.matmul(np.linalg.inv(ZZ),ZY)
        self.w = v[:-1]
        self.b = v[-1]

    def predict(self,x):
        return np.matmul(x,self.w) + self.b

    def RMSE(self,X,Y):
        return np.sqrt(np.mean(np.square(self.predict(X)-Y)))

    def R2(self,X,Y):
        a = np.sum(np.square(self.predict(X)-Y))
        b = np.sum(np.square(Y-np.mean(Y,axis=0)))
        return 1-a/b

    def plotResult(self,X=[],Y=[],xLabel="",yLabel="",fName=""):
        if X.shape[1] != 1: 
            return
        
        fig = plt.figure(figsize=(8,5),dpi=100)

        Xlin = np.array([[0],[np.max(X)]])
        Yplin = self.predict(Xlin)

        plt.plot(X,Y,'.',label="data")
        plt.plot(Xlin,Yplin,'r',label="linear model")
        plt.legend()

        plt.ylim([0,np.max(Y)])
        plt.xlim([0,np.max(X)])
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)

        if len(fName):
            plt.savefig(fName)
        plt.show()