import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N,D = X.shape
    C = W.shape[1]
    for i in xrange(N):
        hepoes = np.dot(X[i], W)
        # print (hepoes.shape)
        scores = np.exp(hepoes)
        # print (scores.shape)
        c = y[i]
        correct_score = scores[c]
        sum_score = np.sum(scores)
        loss -= np.log(correct_score/sum_score)
        dW[:,c] -= X[i].T
        dW += np.dot(X[i].reshape((-1,1)), scores.reshape((1,-1))) / sum_score
    loss /= N
    loss += np.sum(reg * W * W)
    dW /= N
    dW += 2 * reg * W
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N = X.shape[0]

    scores = np.exp(np.dot(X,W))
    sum_scores = np.sum(scores, axis=1)
    correct_scores = scores[np.arange(N),y]

    loss = -np.sum(np.log(correct_scores / sum_scores)) / N
    loss += np.sum(reg * W * W)
    
    scores /= sum_scores.reshape((-1,1))
    scores[np.arange(N),y] -= 1
    dW = np.dot(X.T, scores) / N
    dW += 2 * reg * W

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

