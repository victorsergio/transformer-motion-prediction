
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


transform = MinMaxScaler()
T = 8
S = 12


# create dummy dataset   (100,2)
x1 = np.arange(0, 12)
x2 = np.arange(100,112)   
X = np.vstack((x1,x2)) 
X = np.transpose(X, axes=[1, 0])

X = sliding_window_view(X, 8, 0)

X = np.transpose(X, axes=[0, 2, 1])

print(X.shape)
print(X)

Y = X[1:,:,:]   # this is a copy, for each sequence i on X corresponds sequence i+1 from X on Y
X = X[:-1,:,:]   

print(X.shape)



# create dummy dataset   (100,2)
#x1 = np.arange(0, 40)
#x2 = np.arange(100,140)   
#X = np.vstack((x1,x2)) 
#X = np.transpose(X, axes=[1, 0])

#X = sliding_window_view(X, 20, 0)[::20,:,:]
#X = np.transpose(X, axes=[0, 2, 1])

#Y = X[:,8:,:]
#X = X[:,0:8,:]


#print(X.shape)
#print(Y.shape)
#print(X)
#print(Y)
#sliding_window_view(x, 5)[:, ::2]

#        X = np.transpose(X, axes=[0, 2, 1])

#        self.Y = X[1:,:,:]
#        self.X = X[:-1,:,:]
