
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def sigma_derivative(sigma):
    return sigma*(1-sigma)
    
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
                
#y = np.array([[0, 1, 1, 0]]).T
#y = np.array([[0,0,1,1]]).T
y = np.array([[0,1,0.3,0.5]]).T

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

nextPrintTime = 1
for i in range((1 << 20) + 1):

	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    # how much did we miss the target value?
    l2_error = y - l2
    
    if i >= nextPrintTime:
        nextPrintTime *= 2
        print("Iteration {} Error: {}".format(i, np.mean(np.abs(l2_error))))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*sigma_derivative(l2)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * sigma_derivative(l1)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
