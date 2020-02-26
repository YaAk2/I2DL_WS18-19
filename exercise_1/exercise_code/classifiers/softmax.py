"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    N = X.shape[0]
    D, C = W.shape
 
    for n in range(N):
        x=X[n]
        y_hat = np.dot(x, W)
        exp = np.exp(y_hat - np.max(y_hat))
        sum_exp = np.sum(exp)
        probs = exp/sum_exp
        
        log_probs = np.log(probs)
        loss -= log_probs[y[n]]
     
        for d in range(D):
            for c in range(C):
                dW[d, c] += x[d]*(probs[c]-(c==y[n])) 
    
    loss /= N
    dW /= N
    
    loss += reg*np.sum(W*W)
    dW += reg*W

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################
    
  
             
      
    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    N = X.shape[0]
    D, C = W.shape
    
    y_hat = np.dot(X,W)
    exp = np.exp(y_hat - np.expand_dims(np.max(y_hat, axis = 1), axis = 1))
    sum_exp = np.expand_dims(np.sum(exp, axis = 1), axis = 1)
    probs = exp/sum_exp
  
    log_probs = np.log(probs[list(range(N)), y])
    loss = -np.sum(log_probs, axis = 0)/N
    
    kronecker_delta=np.zeros((N,C))
    kronecker_delta[list(range(N)),y] = 1
    
    dW = np.dot(X.T, (probs - kronecker_delta))/N 
    
    loss += reg*np.sum(W*W)
    dW += reg*W
    
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rate = [3e-6]
    regularization = [0.5]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't t15e much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    
    for lr in learning_rate:
        for r in regularization:
            softmax = SoftmaxClassifier()
            loss_hist = softmax.train(X_train, y_train, 
                                      learning_rate = lr, reg = r, num_iters=20000, verbose=True)
            y_train_pred = softmax.predict(X_train)
            y_pred = softmax.predict(X_val)
            val_accuracy = np.mean(y_val == y_pred)
            if val_accuracy > best_val:
                best_softmax = softmax
                best_val = val_accuracy
            results[(lr, r)] = (np.mean(y_train == y_train_pred), val_accuracy)
            all_classifiers.append(softmax)

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
       
    print('best validation accuracy achieved: %f' % val_accuracy)

    return best_softmax, results, all_classifiers
