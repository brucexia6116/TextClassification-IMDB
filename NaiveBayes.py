import sys
import math
import numpy as np
import pickle
from Eval import Eval

from imdb import IMDBdata
from Vocab import Vocab

class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.data = data # training data
        self.posDict = {}
        self.negDict = {}
        self.totalPos = 0
        self.totalNeg = 0
        self.pos = 0
        self.neg = 0
        self.Train(data.X,data.Y)

    def Train(self, X, Y):
        data = X.data
        indptr = X.indptr
        indices = X.indices

        for i in range(len(Y)):
            #Calculate total positive and negative words
            if Y[i]==1.0:
                self.pos += 1
            else:
                self.neg += 1

            index=indices[indptr[i]:indptr[i+1]]
            for k in range(0,len(index)):
                wordId=index[k]
                v=data[indptr[i]+k]

                #Calculate wordId counts for positive and negative reviews
                if Y[i]==1.0:
                    self.posDict[wordId]=self.posDict.get(wordId,0)+v
                    self.totalPos+=v
                else:
                    self.negDict[wordId] = self.negDict.get(wordId, 0) + v
                    self.totalNeg+=v
        return

    def PredictLabel(self, X):
        data = X.data
        indptr = X.indptr
        indices = X.indices
        vocabSize=self.data.vocab.GetVocabSize()
        Y = []

        #Calculate positive and negative review probabilities
        probPos = (self.pos*1.0)/(self.pos+self.neg)
        probNeg = (self.neg*1.0)/(self.pos+self.neg)

        #Predict class of review
        for i in range(0,len(indptr)-1):
            index = indices[indptr[i]:indptr[i + 1]]
            posClass=math.log(probPos)
            negClass=math.log(probNeg)
            for k in range(0, len(index)):
                wordId = index[k]
                posClass += math.log((self.posDict.get(wordId, 0) + self.ALPHA) / (self.totalPos + (self.ALPHA*vocabSize)))
                negClass += math.log((self.negDict.get(wordId,0) + self.ALPHA)/(self.totalNeg + (self.ALPHA*vocabSize)))

            if posClass>negClass:
                Y.append(+1.0)
            else:
                Y.append(-1.0)
        return Y

    def LogSum(self, logx, logy):
        # log sum exp trick
        if logx>logy:
            m=logx
        else:
            m=logy
        total=math.exp(logx-m)+ math.exp(logy - m)
        return  m+math.log(total)

    def PredictProb(self, test, indexes):
        # predict the probabilities of reviews being positive (first 10 reviews in the test set)
        data = test.X.data
        indptr = test.X.indptr
        indices = test.X.indices
        vocabSize = self.data.vocab.GetVocabSize()
        probPos = (self.pos * 1.0) / (self.pos + self.neg)
        probNeg = (self.neg * 1.0) / (self.pos + self.neg)

        for i in indexes:
            index = indices[indptr[i]:indptr[i + 1]]
            posClass = math.log(probPos)
            negClass = math.log(probNeg)
            for k in range(0, len(index)):
                wordId = index[k]
                posClass += math.log((self.posDict.get(wordId, 0) + self.ALPHA) / (self.totalPos + (self.ALPHA * vocabSize)))
                negClass += math.log((self.negDict.get(wordId, 0) + self.ALPHA) / (self.totalNeg + (self.ALPHA * vocabSize)))

            posProb = posClass-self.LogSum(posClass,negClass)
            negProb = negClass-self.LogSum(posClass,negClass)

            #Print predicted probabilities of respective classes
            if posProb>negProb:
                print test.Y[i], +1.0, math.exp(posProb), test.X_reviews[i]
            else:
                print test.Y[i], -1.0, math.exp(negProb), test.X_reviews[i]


    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X)
        ev = Eval(Y_pred, test.Y)
        return ev.Accuracy()


if __name__ == "__main__":

    traindata = IMDBdata("%s/train" % sys.argv[1])
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)

    #alpha=[0.1,0.5,1.0,5.0,10.0]
    #for a in alpha:

    print "Alpha :",sys.argv[2]
    nb = NaiveBayes(traindata, float(sys.argv[2])) #nb = NaiveBayes(traindata, float(a))
    print nb.Eval(testdata)
    print nb.PredictProb(testdata, range(10))
    print "################################################################################################"