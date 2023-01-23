import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self,X,Y,priors):
        self.X = X
        self.Y = Y
        self.wordDict = list(np.unique(np.concatenate(self.X)))
        self.wordNums = self.countWords(self.X)
        self.priors = priors

    def countWords(self,X):
        countWordsAll =[]
        for words in X:
            countWords = np.zeros(len(self.wordDict))
            for word in words:
                cnt = self.wordDict.count(word)
                if cnt > 0:
                    countWords[self.wordDict.index(word)]+=cnt
            countWordsAll.append(countWords)
        return np.array(countWordsAll)

    def train(self):
        wordNumsCat = [np.sum(self.wordNums[self.Y == y],axis=0) for y in np.unique(self.Y)]
        wordNumsCat = np.array(wordNumsCat)
        self.wordL = wordNumsCat/np.sum(wordNumsCat,axis=1,keepdims=True)

    def predict(self,X):
        wordNums = self.countWords(X)
        sentenceL = [np.product(self.wordL[ind]**wordNums,axis=1) for ind in range(len(np.unique(self.Y)))]
        sentenceL = np.array(sentenceL)
        sentenceP = sentenceL.T * self.priors
        predict = np.argmax(sentenceP,axis=1)
        return predict

    def accuracy(self,X,Y):
        return np.sum(self.predict(X)-Y.T == 0)/len(Y)

    def writeResult2CSV(self,X,Y,fName="../results/sentimental_results.csv"):
        P = self.predict(X)
        df = pd.DataFrame(np.array([Y,P,X]).T,columns=['gt','predict','sentence'],index=np.arange(len(X)))
        df.to_csv(fName,index=False)
    