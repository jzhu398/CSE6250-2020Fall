# Do not use anything outside of the standard distribution of python
# when implementing this class
import math

class LogisticRegressionSGD:
    """
    Logistic regression with stochastic gradient descent
    """

    def __init__(self, eta, mu, n_feature):
        """
        Initialization of model parameters
        """
        self.eta = eta
        self.weight = [0.0] * n_feature
        self.mu = mu

    def fit(self, X, y):
        """
        Update model using a pair of training sample
        """
        ## calculate y based on the current w and x
        y_pred = math.fsum((self.weight[f]*v for f, v in X)) 
        ## calcuate signoid function of y
        if y_pred < 0:
            sig = 1.0 - 1.0/(1.0 + math.exp(y_pred))
        else:
            sig = 1.0/(1.0 + math.exp(-y_pred))
        
        for f,v in X:
            self.weight[f] = self.weight[f]+self.eta*(v*(y-sig))-2*self.eta*self.mu*self.weight[f]
            
        pass

    def predict(self, X):
        """
        Predict 0 or 1 given X and the current weights in the model
        """

        return 1 if predict_prob(X) > 0.5 else 0

    def predict_prob(self, X):
        """
        Sigmoid function
        """
        y_pred = math.fsum((self.weight[f]*v for f, v in X))
        if y_pred <0:
            sig = 1.0-1.0/(1.0+math.exp(y_pred))
        else:
            sig = 1.0 / (1.0 + math.exp(-y_pred))
        return sig