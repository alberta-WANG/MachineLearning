import numpy as np
import matplotlib.pylab as plt

class LDA:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.dNum = X.shape[0]
        self.xDim = X.shape[1]

        self.Xneg = X[Y[:,0]==-1]
        self.Xpos = X[Y[:,0]==1]

        self.m = np.mean(self.X,axis=0,keepdims=True)
        self.mNeg = np.mean(self.Xneg,axis=0,keepdims=True)
        self.mPos = np.mean(self.Xpos,axis=0,keepdims=True)

    def train(self):
        Sinter = np.matmul((self.mNeg-self.mPos).T,self.mNeg-self.mPos)
        Xneg = self.Xneg - self.mNeg
        Xpos = self.Xpos - self.mPos
        Sintra = np.matmul(Xneg.T,Xneg) + np.matmul(Xpos.T,Xpos)
        [L,V] = np.linalg.eig(np.matmul(np.linalg.inv(Sintra),Sinter))
        self.w = V[:,[np.argmax(L)]]

    def predict(self,x):
        return np.sign(np.matmul(x-self.m,self.w))

    def accuracy(self,x,y):
        return np.sum(self.predict(x)==y)/len(x)

    def plotModel2D(self,X=[],Y=[],xLabel="",yLabel="",title="",fName=""):
        if X.shape[1] != 2: return
    
        fig = plt.figure(figsize=(6,4),dpi=100)
        
        # 最小と最大の点
        X1 = np.arange(np.min(X[:,0]),np.max(X[:,0]),(np.max(X[:,0]) - np.min(X[:,0]))/100)
        X2 = (np.matmul(self.m,self.w)[0] - X1*self.w[0])/self.w[1]

        # データと線形モデルのプロット
        plt.plot(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],'cx',label="category:-1")
        plt.plot(X[Y[:,0]==1,0],X[Y[:,0]==1,1],'m.',label="category:+1")
        plt.plot(self.m[0,0],self.m[0,1],'ko',label="average")
        plt.plot(X1,X2,'r-',label="f(x)")
        
        # 各軸の範囲とラベルの設定
        plt.xlim([np.min(X[:,0]),np.max(X[:,0])])
        plt.ylim([np.min(X[:,1]),np.max(X[:,1])])
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.legend()

        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
            plt.show()