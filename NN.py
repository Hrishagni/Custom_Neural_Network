from numpy.lib.function_base import select
import pandas as pd
import numpy as np

def sigmoid(z):
    a = ( 1 / ( 1 + np.exp(-z) ) )
    return a

def sigmoidPrime(z):
    a = sigmoid(z) * (1 - sigmoid(z) ) 
    return a

def relu(z):  
  return np.maximum(z, 0)

def relu_prime(z):  
  return (z>0).astype(z.dtype)

def tanh(Z):
    return np.tanh(Z)

def tanh_prime(Z):
  return 1-(tanh(Z)**2)

class NeuralNetwork:
    
    def __init__(self , filePath, learningRate,hidden_units):  

        np.random.seed(1)        
        dataset = pd.read_csv(filePath)
        self.learningRate=learningRate
        self.hidden_units=hidden_units
        self.x_train = dataset.iloc[:,0:3]
        self.y_train = np.array(dataset.iloc[:,3]).reshape(self.x_train.shape[0],1)
        self.m = len(self.x_train) 
        
        self.w1 = np.random.randn(self.x_train.shape[1],self.hidden_units) # w1=3,4
        self.w2 = np.random.randn(self.hidden_units,1) # w2=4,1        
        self.b1 = np.zeros((self.x_train.shape[0],self.hidden_units))
        self.b2 = np.zeros((self.x_train.shape[0],1))

        
    def forwardProp(self):
        self.z1 = np.dot(self.x_train , self.w1) + self.b1  # z1=8,4        
        self.a1 = relu(self.z1) # a1=8,4
        
        self.z2 = np.dot(self.a1 , self.w2) + self.b2 # z2=8,1
        self.a2 = relu(self.z2) # a2=8,1

        #print(self.z1.shape,self.a1.shape,self.z2.shape,self.a2.shape,self.w1.shape,self.w2.shape)
           

    def backwardProp(self):        
        self.dz2 = self.a2-self.y_train        
        self.dw2 = (1/self.m) * np.dot(np.transpose(self.a1),self.dz2)
        self.db2 = (1/self.m) * np.sum(self.dz2,axis=1,keepdims=True)
        self.dz1 = np.dot(self.dz2,np.transpose(self.w2)) * relu_prime(self.z1)
        self.dw1 = (1/self.m) * np.dot(np.transpose(self.x_train),self.dz1)
        self.db1 = (1/self.m) * np.sum(self.dz1,axis=1,keepdims=True)

    def updateWeights(self):
        
        self.w1 = self.w1 - self.learningRate * self.dw1
        self.w2 = self.w2 - self.learningRate * self.dw2
        self.b1 = self.b1 - self.learningRate * self.b1
        self.b2 = self.b2 - self.learningRate * self.b2
    
    def costFunc(self):   
        return np.sqrt((1/self.m)*np.sum(np.square(self.y_train-self.a2)))  
        # return np.squeeze(-(1./self.m)*np.sum(np.multiply(self.y_train, np.log(self.a2))+np.multiply(np.log(1-self.a2), 1-self.y_train)))

if __name__=="__main__":    
    cost=[]
    epochs=2000
    learningRate = 0.01
    hidden_units = 10
    obj=NeuralNetwork(r'C:\Hrishagni\NN_Scratch\3-input-OR.csv',learningRate,hidden_units)  

    for e in range(epochs+1):
        obj.forwardProp()
        loss=obj.costFunc()
        cost.append(loss)
        obj.backwardProp()
        obj.updateWeights()
        if e%(epochs/10)==0:
            print(f"Epoch {e}\tLoss {loss:.4f}")

    obj.z1 = np.dot(obj.x_train , obj.w1) + obj.b1  # z1=8,4        
    obj.a1 = relu(obj.z1) # a1=8,4
    
    obj.z2 = np.dot(obj.a1 , obj.w2) + obj.b2 # z2=8,1
    obj.a2 = relu(obj.z2) # a2=8,1


    
    
    
    
    

        
        

