import sys

import numpy as np
from Eval import Eval

from imdb import IMDBdata

class Perceptron:
    def __init__(self, X, Y, N_ITERATIONS):
        self.N_ITERATIONS = N_ITERATIONS
        self.weights = {w: 0.0 for w in range(X.shape[1])}
        self.b=0.0
        self.traindataX=X
        self.traindataY=Y
        #self.Train(X,Y)


    def ComputeAverageParameters(self):
        #Compute average parameters
        X=self.traindataX
        Y=self.traindataY
        data = X.data
        indptr = X.indptr
        indices = X.indices
        self.weights = {w: 0.0 for w in range(X.shape[1])}
        cachedWeights = {u: 0.0 for u in range(X.shape[1])}
        self.b = 0.0
        cachedBias = 0.0
        c=1

        for n in range(self.N_ITERATIONS):
            for i in range(len(Y)):
                index = indices[indptr[i]:indptr[i + 1]]
                a=self.b
                for k in range(0, len(index)):
                    wordId = index[k]
                    x = data[indptr[i] + k]
                    a+=(self.weights.get(wordId, 0.0) * x)

        #Update average parameters
                if (Y[i] * a) <=0:
                    for k in range(0, len(index)):
                        wordId = index[k]
                        x = data[indptr[i] + k]
                        self.weights[wordId] = self.weights.get(wordId, 0.0) + (Y[i] * x)
                        cachedWeights[wordId] = cachedWeights.get(wordId, 0.0) + (Y[i] * c * x)
                    self.b += Y[i]
                    cachedBias += (Y[i] * c)
                c += 1

        self.b-=(cachedBias/c)
        self.weights = {key: self.weights[key] - (cachedWeights.get(key, 0.0)/c) for key in self.weights.keys()}
        return

    def Train(self, X, Y):
        #Estimate perceptron parameters- weights and bias
        data = X.data
        indptr = X.indptr
        indices = X.indices
        c=0
        for n in range(self.N_ITERATIONS):
            c+=1
            for i in range(len(Y)):
                index = indices[indptr[i]:indptr[i + 1]]
                a=self.b
                for k in range(0, len(index)):
                    wordId = index[k]
                    x = data[indptr[i] + k]
                    a+=(self.weights.get(wordId,0.0)*x)

        #Update weights and bias
                if Y[i]*a <=0:
                    for k in range(0, len(index)):
                        wordId = index[k]
                        x = data[indptr[i] + k]
                        self.weights[wordId]=self.weights.get(wordId,0.0)+(Y[i]*x)
                    self.b+=Y[i]

        return

    def Predict(self, X):
        #Predict label of reviews by perceptron classification
        data = X.data
        indptr = X.indptr
        indices = X.indices
        Y=[]

        for i in range(0,len(indptr)-1):
            index = indices[indptr[i]:indptr[i + 1]]
            a = self.b
            for k in range(0, len(index)):
                wordId = index[k]
                v = data[indptr[i] + k]
                a += (self.weights.get(wordId, 0.0) * v)

            if a>=0:
                Y.append(+1.0)
            else:
                Y.append(-1.0)

        return Y

    def Eval(self, X_test, Y_test):
        Y_pred = self.Predict(X_test)
        ev = Eval(Y_pred, Y_test)
        return ev.Accuracy()

    def getWords(self,vocab):
        values=self.weights.values()
        indices=np.argsort(values)
        c=1
        print "---------Positive words---------"
        for i in xrange(indices.size-1,indices.size-21,-1):
            print c,' ',vocab.GetWord(indices[i]),' ',self.weights.get(indices[i])
            c+=1

        c=1
        print "---------Negative words---------"
        for i in indices[0:20]:
            print c,' ',vocab.GetWord(i),' ',self.weights.get(i)
            c+=1


if __name__ == "__main__":
    train = IMDBdata("%s/train" % sys.argv[1])
    test  = IMDBdata("%s/test" % sys.argv[1], vocab=train.vocab)

    #epochs = [1,10,50,100,500,1000]
    #for e in epochs:
    ptron = Perceptron(train.X, train.Y,int(sys.argv[2]))

    print "Epoch ",int(sys.argv[2])
    print "######### Vanilla Perceptron #########"
    ptron.Train(train.X,train.Y)
    print ptron.Eval(test.X, test.Y)
    print ptron.getWords(train.vocab)

    print "######### Average Perceptron #########"
    ptron.ComputeAverageParameters()
    print ptron.Eval(test.X, test.Y)
    print ptron.getWords(train.vocab)