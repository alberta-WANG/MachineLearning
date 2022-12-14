import numpy as np
import matplotlib.pylab as plt

class logisticRegression:
    # 1. setting training data and initializing model parameter
    # X: input data(number of samples * features of a sample)(type:numpy.ndarray)
    # Y: output data(number of samples * 1)(type:numpy.ndarray)
    def __init__(self,X,Y):
        # setting training data
        self.X = X
        self.Y = Y
        self.dNum = X.shape[0] # number of samples
        self.xDim = X.shape[1] # features of a sample

        # add 1 as a feature at the end of every samples of X
        self.Z = np.concatenate([self.X,np.ones([self.dNum,1])],axis=1)
        
        # setting model parameter
        self.w = np.random.normal(size=[self.xDim,1])
        self.b = np.random.normal(size=([1,1]))

        # approximate value of ln(0)
        self.smallV = 10e-8

    # 2. updating parameter with decreasing gradient algorithm
    # alpha: learning rate(scale)
    def update(self,alpha=0.1):
        # determining error between prediction and Y
        P = self.predict(self.X)
        error = P-self.Y

        # updating parameter
        grad = 1/self.dNum * np.matmul(self.Z.T,error)
        v = np.concatenate([self.w,self.b],axis=0)
        v -= alpha * grad

        # determining parameter w,b
        self.w = v[:-1]
        self.b = v[[-1]]

    # 3. predicting y with linear and sigmoid function
    def predict(self,x):
        f_x = np.matmul(x,self.w) + self.b
        res = 1/(1+np.exp(-f_x))
        return res

    # 4. loss function with cross entropy
    def CE(self,X,Y):
        P = self.predict(X)
        res = -np.mean(Y * np.log(P+self.smallV) + (1-Y) * np.log(1-P+self.smallV))
        return res

    # 5. accuracy
    def accuracy(self,X,Y,thre=0.5):
        P = self.predict(X)
        P[P>thre] = 1
        P[P<=thre] = 0
        accuracy = np.mean(Y==P)
        return accuracy

    # 6. 真値と予測値のプロット（入力ベクトルが1次元の場合）
    # X: 入力データ（次元数×データ数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    # xLabel: x軸のラベル（文字列）
    # yLabel: y軸のラベル（文字列）
    # fName: 画像の保存先（文字列）
    def plotModel1D(self,X=[],Y=[],xLabel="",yLabel="",fName=""):
        fig = plt.figure(figsize=(6,4),dpi=100)

        # 予測値
        P = self.predict(X)

        # 真値と予測値のプロット
        plt.plot(X,Y,'b.',label="real value")
        plt.plot(X,P,'r.',label="predict value")
        
        # 各軸の範囲とラベルの設定
        plt.yticks([0,0.5,1])
        plt.ylim([-0.1,1.1])
        plt.xlim([np.min(X),np.max(X)])
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid()
        plt.legend()

        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
            plt.show()
    #-------------------

    #-------------------
    # 7. 真値と予測値のプロット（入力ベクトルが2次元の場合）
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    # xLabel: x軸のラベル（文字列）
    # yLabel: y軸のラベル（文字列）
    # title: タイトル（文字列）
    # fName: 画像の保存先（文字列）
    def plotModel2D(self,X=[],Y=[],xLabel="",yLabel="",title="",fName=""):
        fig = plt.figure(figsize=(6,4),dpi=100)
        #plt.close()
        
        # 真値のプロット（クラスごとにマーカーを変更）
        plt.plot(X[Y[:,0]==0,0],X[Y[:,0]==0,1],'cx',label="label0")
        plt.plot(X[Y[:,0]==1,0],X[Y[:,0]==1,1],'m.',label="label1")

        # 予測値のメッシュの計算
        X1,X2 = plt.meshgrid(plt.linspace(np.min(X[:,0]),np.max(X[:,0]),50),plt.linspace(np.min(X[:,1]),np.max(X[:,1]),50))
        Xmesh = np.hstack([np.reshape(X1,[-1,1]),np.reshape(X2,[-1,1])])
        Pmesh = self.predict(Xmesh)
        Pmesh = np.reshape(Pmesh,X1.shape)

        # 予測値のプロット
        CS = plt.contourf(X1,X2,Pmesh,cmap="bwr",alpha=0.3,vmin=0,vmax=1)

        # カラーバー
        CB = plt.colorbar(CS)
        CB.ax.tick_params(labelsize=14)

        # 各軸の範囲とラベルの設定
        plt.xlim([np.min(X[:,0]),np.max(X[:,0])])
        plt.ylim([np.min(X[:,1]),np.max(X[:,1])])
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.legend()

        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
            plt.show()
    #-------------------

    #------------------- 
    # 8. 学習と評価損失のプロット
    # trEval: 学習の損失
    # teEval: 評価の損失
    # yLabel: y軸のラベル（文字列）
    # fName: 画像の保存先（文字列）
    def plotEval(self,trEval,teEval,ylabel="loss",fName=""):
        fig = plt.figure(figsize=(6,4),dpi=100)
        
        # 損失のプロット
        plt.plot(trEval,'b',label="train")
        plt.plot(teEval,'r',label="estimate")
        
        # 各軸の範囲とラベルの設定
        plt.xlabel("step")
        plt.ylabel(ylabel)
        plt.ylim([0,1.1])
        plt.legend()
        
        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
            plt.show()
    #-------------------