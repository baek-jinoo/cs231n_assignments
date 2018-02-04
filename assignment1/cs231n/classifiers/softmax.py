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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_data = X.shape[0]
  for data_index in xrange(num_data):
    scores = X[data_index].dot(W)
    denominator = 0
    numerator = 0
    for class_index in xrange(num_classes):
      exponent = np.exp(scores[class_index])
      denominator += exponent
      if class_index == y[data_index]:
        numerator = exponent
    loss += -np.log(numerator/denominator)
    dW += -(denominator/numerator)
  loss /= num_data
  dW /= num_data
  loss += reg * np.sum(W * W)
  dW -= 2 * reg * W
  
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W) # (500, 10)
  maxes = np.max(scores, axis=1)
  maxes = np.expand_dims(maxes, 1)
  scores -= maxes
  exps = np.exp(scores)
  label_scores = exps[np.arange(len(y)), y]
  sums = np.sum(exps, axis=1)
  label_probs = label_scores / sums
  print("probs", label_probs)
  label_logs_probs = -np.log(label_probs)
  print("logs", label_logs_probs)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

