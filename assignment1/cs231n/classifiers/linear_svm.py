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
        dW[:,y[i]] -= X[i]
        dW[:,j] += X[i]
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW -= 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
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
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  dot_product = np.dot(X,W) # (500, 3073) * (3073, 10) => (500, 10)
  label_scores = dot_product[np.arange(len(y)), y] # use label vector to get scores out (500, )
  pre_max = dot_product - np.expand_dims(label_scores, 1) + 1 # evaluate scores for non label values => (500, 10)
  pre_max[np.arange(len(dot_product)), y] = 0 # set label scores to zero => (500, 10)
  scores = np.maximum(pre_max, 0) # only care about scores above zero => (500, 10)
  loss = np.sum(scores) / len(scores) # element-wise divide by N => ()
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  scores_bool = scores > 0 # turn all turned on gradient to True
  gradient_multiple = scores_bool * 1. # turn Trues into 1.
  num_negate_classes_for_label = np.sum(gradient_multiple, axis=1) # count the number of negative gradients
  gradient_multiple[np.arange(len(y)), y] = -num_negate_classes_for_label # place the negative gradient counts for target label
  dW = np.dot(X.T, gradient_multiple) / len(y) # divide by count for the last step of the computation graph via chain rule
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
