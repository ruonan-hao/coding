import scipy.io as sio
import numpy as np


class NeuralNets(object):

    def __init__(self,hidden_units=200, mu=0,sd=0.1,nrow=784,label_n=26):
        self.mu = mu
        self.sd = sd
        self.nrow = nrow
        self.label_n = label_n
        self.hidden_units = hidden_units
        self.V = self._initialize(self.hidden_units,self.nrow +1)
        self.W = self._initialize(self.label_n,self.hidden_units+1)


    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))


    def _cross_entropy(self,y, z):
        step1 = y.T * np.log(z)
        step2 = (1 - y.T) * np.log(1 - z)
        final = -np.sum(step1 + step2)
        return final


    def _initialize(self,n, m):
        return np.random.normal(self.mu, self.sd, (n, m))


    def _forward(self,images):
        a1 = images
        # hidden_layer
        self.z2 = self.V.dot(a1.T)
        a2 = np.tanh(self.z2)
        d = a1.shape[0]
        self.a2 = np.vstack((a2, np.ones(d)))
        # output_layer
        z3 = self.W.dot(self.a2)
        z = self._sigmoid(z3)
        return z

    def _backpropagation(self,X,y, z):
        gradient_H = self.W.T.dot(z - y.T)
        gradient_W = (z - y.T).dot(self.a2.T)
        gradient_V = np.multiply(gradient_H, (1 - self.a2 * self.a2)).dot(X)
        return gradient_V,gradient_W



    def trainNeuralNetwork(self,images, labels,X_val,y_val,Iter=100, learning_rate1=0.001,\
                            learning_rate2=0.01, batch = 1):
        """
        images:training images(X)
        labels:training labels(y)
        params:hyperparameters learning_rate, weight_decay

        return weight matrix V, W
        """
         #images.shape[0] * epoch
        i = 0
        accuracy,loss = [],[]
        while (i <= Iter):
            print('Iter:', i)
            # forward pass
            index = np.random.randint(0, images.shape[0], batch)
            sub_X = images[index]
            sub_y = labels[index]
            pred = self._forward(sub_X)

            # backward pass
            gradient_V, gradient_W = self._backpropagation(sub_X,sub_y,pred)

            # compute accuracy and loss
            val_pred = self._forward(X_val)
            y_pred = np.argmax(val_pred,0) + 1
            print(y_pred)
            y_hat = np.argmax(y_val,1) + 1
            print(y_hat)
            acc = np.sum(y_hat == y_pred)/y_val.shape[0]
            print(acc)
            accuracy.append(acc)
            loss.append(self._cross_entropy(y_val,val_pred))
            self.W = self.W - learning_rate2 * gradient_W
            self.V = self.V - learning_rate1 * gradient_V[range(gradient_V.shape[0] - 1)]

            i += 1
        return accuracy, loss

    def predictNeuralNetwork(self,test_images):
        """
        images: test images
        """
        pred = self._forward(test_images)
        return np.argmax(pred,0) + 1
