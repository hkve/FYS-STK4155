'''
This is very messy, I have just done the first things that have come to mind, and I am rusty...
'''
import numpy as np
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )



class TrainingFacility: # working title
    def __init__(self, model, data):
        '''
        model : a Model class to undergo training etc.
        data  : array of data in shape (y, *X), where *X denotes columns of X.
        '''
        self.model = model
        self.data = data

        self.scaled_data = None

        self.scalers_ = {
            "Standard" : self.standard_scaler_
        }

    def unpack_data_(self, data):
        '''
        Unpacks data array into y and X.
        NB. This is a mess made from being to fancy with data-packing...
        '''
        return data[:,0], data[:,1:]

    def pack_data_(self, y, X):
        '''
        Packs y, X into shape (y, *X)
        NB. See unpack data...
        '''
        return np.array([(yi, *xi) for yi, xi in zip(y, X)])

    def scale_data(self, scheme="Standard"):
        '''
        Scales the data according to scheme

        NB. Schemes not implemented yet, deafults to Standard scaling of y, x to be N(1,0)-distributed.
        '''
        y, X = self.unpack_data_(self.data)
        ynew, Xnew = self.scalers_[scheme](y, X)
        self.scaled_data = self.pack_data_(ynew, Xnew)

        return self

    def train_test_split(self, train_test=2/3):
        '''
        Splits the data into training data and test data according to train_test-ratio
        '''
        size = self.data.shape[0]
        shuffled_data = self.data.copy(); np.random.shuffle(shuffled_data)
        training_data, test_data = shuffled_data[:int(size*train_test)], shuffled_data[int(size*train_test):]
        return training_data, test_data

    def fit_training_data(self, train_test=2/3):
        '''
        Fits the model to training data.
        NB. Not sure just how I want to implement this bit.
        '''
        training_data, test_data = self.train_test_split(train_test=train_test)
        y_training, X_training = self.unpack_data_(training_data)
        fit_model = self.model(method="INV").fit(X_training, y_training)
        return fit_model

    def standard_scaler_(self, y, X):
        '''
        Scales y, *X to be N(0, 1)-distributed.
        '''
        ynew = (y - np.mean(y)) / np.where(np.std(y)==0, 1, np.std(y))
        Xnew = (X - np.mean(X, axis=0)) / np.where(np.std(X, axis=0)==0, 1, np.std(X, axis=0))
        return ynew, Xnew



if __name__ == '__main__':
    from sknotlearn.linear_model import LinearRegression
    x = np.random.uniform(0, 1, size=100)
    X = np.array([np.ones_like(x), x, x**2]).T
    beta = np.array([1, -2, 3])
    y = np.exp(x*x) + 2*np.exp(-2*x) + 0.1*np.random.randn(x.size)
    data = np.array([(yi, 1, xi, xi**2) for yi, xi in zip(y,x)])


    test = TrainingFacility(LinearRegression, data)
    test.scale_data()
    fit = test.fit_training_data()
    print(f"MSE score : {np.mean((fit.predict(X)-y))**2}")

    import matplotlib.pyplot as plt
    plt.scatter(x, y)
    x.sort()
    ypredict = fit.predict(np.array([np.ones_like(x), x, x**2]).T)
    plt.plot(x, ypredict)
    plt.show()