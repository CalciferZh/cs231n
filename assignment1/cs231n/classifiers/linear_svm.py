import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:].T
        dW[:,y[i]] -= X[i,:].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                              #
  # Compute the gradient of the loss function and store it dW.           #
  # Rather that first computing the loss and then computing the derivative,  #
  # it may be simpler to compute the derivative at the same time that the    #
  # loss is being computed. As a result you may need to modify some of the  #
  # code above to compute the gradient.                          #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                              #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                       #
  #############################################################################
  (N,D) = X.shape
  scores = np.dot(X,W)
  correct_scores = scores[np.arange(N),y]
  margin = scores - correct_scores.reshape((-1,1)) + 1
  margin[np.arange(N), y] = 0
  # using mask here might be a little bit slower
  # but the mask can be reused later
  mask = np.zeros(margin.shape)
  mask = margin > 0
  loss = np.sum(margin * mask) / N
  # margin = np.maximum(np.zeros(margin.shape), margin)
  # loss = np.sum(margin) / N
  loss += reg * np.sum(W * W)
  pass
  #############################################################################
  #                             END OF YOUR CODE           #
  #############################################################################


  #############################################################################
  # TODO:                                              #
  # Implement a vectorized version of the gradient for the structured SVM    #
  # loss, storing the result in dW.                            #
  #                                                  #
  # Hint: Instead of computing the gradient from scratch, it may be easier  #
  # to reuse some of the intermediate values that you used to compute the   #
  # loss.                                              #
  #############################################################################
  
  grad_w = np.zeros(mask.shape)
  grad_w[mask] = 1

  hit = np.sum(mask, axis=1)
  grad_w[np.arange(N),y] = -hit[np.arange(N)]
  dW = np.dot(grad_w.T, X).T
  dW /= N
  dW += 2 * reg * W

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
