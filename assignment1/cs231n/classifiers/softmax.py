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
  X -= np.max(X, axis=0)
  num_classes = W.shape[1]
  num_data = X.shape[0]

  for data_index in xrange(num_data):
    scores = X[data_index].dot(W)
    denominator = 0
    numerator = 0
    exponents = {}

    # calculate loss
    for class_index in xrange(num_classes):
      exponent = np.exp(scores[class_index])
      exponents[class_index] = exponent
      denominator += exponent
      if class_index == y[data_index]:
        numerator = exponent

    loss += -np.log(numerator/denominator)

    # calculate gradient
    for class_index in xrange(num_classes):
      exponent = exponents[class_index]
      score = scores[class_index]
      grad_e = np.exp(score)
      grad_log = (denominator / numerator)

      if class_index == y[data_index]:
        grad_product = 1/denominator
        dW[:, class_index] += X[data_index] * grad_e * grad_product * grad_log

      grad_reciprocal = -1 / (denominator**2)
      grad_product = numerator
      dW[:, class_index] += X[data_index] * grad_e * grad_reciprocal * grad_product * grad_log

  loss /= num_data
  dW /= num_data
  dW *= -1

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  
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

  # Shift data down (reduce magnitude) to improve numerical stability
  maxes = np.max(scores, axis=1)
  maxes = np.expand_dims(maxes, 1)
  scores -= maxes

  exps = np.exp(scores)
  label_exps = exps[np.arange(len(y)), y]
  sums = np.sum(exps, axis=1)
  probs = label_exps / sums
  ln_probs = -np.log(probs)
  softmax_loss = np.sum(ln_probs) / len(y)
  loss = softmax_loss + reg * np.sum(W * W)

  def dln():
    grad_ln = 1 / probs
    grad_negative = -1
    grad_average = 1 / len(y)
    
    return grad_average * grad_negative * grad_ln
    
  def numerator_gradient():
    grad_exp = label_exps
    grad_label_exps_multiply = 1 / sums

    dExpPreDot = dln() * grad_label_exps_multiply * grad_exp
    dExp = np.zeros_like(scores)
    dExp[np.arange(len(y)), y] = dExpPreDot
    return np.dot(X.T, dExp)

  def denominator_gradient():
    grad_exp = exps
    grad_flip = -1 / (sums * sums)
    grad_label_exps_multiply = label_exps

    dSum = dln() * grad_label_exps_multiply * grad_flip
    dExp = grad_exp * np.expand_dims(dSum, 1)
    return np.dot(X.T, dExp)

  dW += denominator_gradient()
  dW += numerator_gradient()

  # Regularization gradient
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

