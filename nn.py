

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def sigma_derivative(sigma):
    return sigma*(1-sigma)
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset 
# easy case , just learn that Y = X[0]         
#y = np.array([[0,0,1,1]]).T
# hard case, simple net does not learn it perfectly
y = np.array([[0,1,0.3,0.5]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# Hidden layer, 3 neurons
# Initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

nextPrintTime = 1
for i in range((1 << 20) + 1):
    # forward propagation
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * sigma_derivative(l1)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)
    
    if i >= nextPrintTime:
        print("Iteration {}\nl1\n{}\nl1_error\n{}".format(i, l1, l1_error))
        print("Mean square error:\n{}".format(np.mean((y-l1)**2)))
        nextPrintTime *= 2
