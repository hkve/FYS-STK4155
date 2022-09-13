'''
This is very messy, I have just done the first things that have come to mind, and I am rusty...
'''
from sknotlearn.linear_model import LinearRegression
import numpy as np
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )


class Data:
    '''
    Class for storing and handling (y, X) data
    '''
    def __init__(self, y, X):
        self.y = np.array(y)
        self.X = np.array(X)
        self.n_features = X.shape[-1]+1

        self.unscale = lambda self : self

    def __len__(self):
        return 

    def __getitem__(self, key):
        try:
            i, j = key
            if i == 0:
                return self.y[j]
            elif i > 0 and i < self.X.shape[1]:
                return self.X[j, i]
            else:
                raise IndexError(f"Index {i} out of bounds for data of with {self.n_features} features.")
        except ValueError:
            if i==0:
                return self.y
            elif i > 0 and i < self.X.shape[1]:
                return self.X[:,i]
            else:
                raise IndexError(f"Index {i} out of bounds for data of with {self.n_features} features.")


    def __iter__(self):
        self.i = 0
        yield Data(self.y[self.i], self.X[self.i])

    def __next__(self):
        self.i += 1

    def __str__(self):
        return str(np.concatenate(([self.y], [*self.X.T])).T) # messy, I know

    def __add__(self, other):
        if type(other) in (int, float, np.int64, np.float64):
            return Data(self.y+other, self.X+other)
        elif type(other) == np.ndarray:
            try:
                return Data(self.y+other[:,0], self.X+other[:,1:])
            except IndexError:
                return Data(self.y+other[0], self.X+other[1:])
        elif type(other) == Data:
            return Data(self.y+other.y, self.X+other.X)
        else:
            raise TypeError(f"Addition not implemented betwee Data and {type(other)}")
    
    def __sub__(self, other):
        if type(other) in (int, float, np.int64, np.float64):
            return Data(self.y-other, self.X-other)
        elif type(other) == np.ndarray:
            try:
                return Data(self.y-other[:,0], self.X-other[:,1:])
            except IndexError:
                return Data(self.y-other[0], self.X-other[1:])
        elif type(other) == Data:
            return Data(self.y-other.y, self.X-other.X)
        else:
            raise TypeError(f"Subtraction not implemented betwee Data and {type(other)}")

    def __mul__(self, other):
        if type(other) in (int, float, np.int64, np.float64):
            return Data(self.y*other, self.X*other)
        elif type(other) == np.ndarray:
            try:
                return Data(self.y*other[:,0], self.X*other[:,1:])
            except IndexError:
                return Data(self.y*other[0], self.X*other[1:])
        elif type(other) == Data:
            return Data(self.y*other.y, self.X*other.X)
        else:
            raise TypeError(f"Multiplication not implemented betwee Data and {type(other)}")

    def __truediv__(self, other):
        if type(other) in (int, float, np.int64, np.float64):
            return Data(self.y/other, self.X/other)
        elif type(other) == np.ndarray:
            try:
                return Data(self.y/other[:,0], self.X/other[:,1:])
            except IndexError:
                return Data(self.y/other[0], self.X/other[1:])
        elif type(other) == Data:
            return Data(self.y/other.y, self.X/other.X)
        else:
            raise TypeError(f"Division not implemented between Data and {type(other)}")


    def unpack(self):
        return self.y, self.X

    def train_test_split(self, ratio=2/3, random_seed=None):
        '''
        Splits the data into training data and test data according to train_test-ratio
        '''
        np.random.seed(seed=random_seed) # allows control over random seed

        size = self.y.size
        split_idx = int(size*ratio)
        shuffled_idxs = np.arange(size); np.random.shuffle(np.arange(size))
        training_idxs = shuffled_idxs[:split_idx]
        test_idxs = shuffled_idxs[split_idx:]
        training_data = Data(self.y[training_idxs], self.X[training_idxs])
        test_data = Data(self.y[test_idxs], self.X[test_idxs])
        return training_data, test_data

    def scale(self, scheme="None"):
        return self.scalers_[scheme](self)

    def none_scaler_(self, data):
        '''
        Does not scale y, *X.
        '''
        return data

    def standard_scaler_(self, data):
        '''
        Scales y, *X to be N(0, 1)-distributed.
        '''
        data = np.concatenate(([data.y], [*data.X.T])).T
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std = np.where(data_std != 0, data_std, 1) # sets unvaried data-columns to 0
        data_scaled = (data - data_mean) / data_std

        scaled_data = Data(data_scaled[:,0], data_scaled[:,1:])
        scaled_data.unscale = lambda data : data*data_std + data_mean

        return scaled_data
    
    scalers_ = {
        "None" : none_scaler_,
        "Standard" : standard_scaler_
    }



class TrainingFacility: # working title
    def __init__(self, model, data):
        '''
        model : a Model class to undergo training etc.
        data  : an instance of the Data class
        '''
        self.model = model
        self.data = data

        self.isFit = False
        self.fitmodel = None
        self.scaled_data = None
        self.unscale_data = lambda scaled_data : None

    def fit_training_data(self, scaler="None", ratio=2/3, random_seed=None):
        '''
        Fits the model to training data.
        NB. Not sure just how I want to implement this bit.
        '''
        scaled_data = self.data.scale(scheme=scaler)
        self.training_data, self.test_data = self.data.train_test_split(ratio=ratio, random_seed=random_seed)
        y_training, X_training = self.training_data.unpack()
        self.fit_model = self.model(method="INV").fit(X_training, y_training)
        self.isFit = True
        return self.fit_model

    def predict_test_data(self):
        if self.isFit:
            X_test = self.test_data.unpack()[1]
            y_predict = self.fit_model.predict(X_test)
            return y_predict
        else:
            raise Exception("Cannot make prediction, model has not yet been fitted to data.")


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.random.uniform(0, 1, size=100)
    X = np.array([np.ones_like(x), x, x**2]).T
    y = np.exp(x*x) +2*np.exp(-2*x) + 0.1*np.random.randn(x.size)
    testdata = Data(y, X)

    tester = TrainingFacility(LinearRegression, testdata)

    test_fit = tester.fit_training_data()
    ytest, Xtest = tester.test_data.unpack()
    ypredict = test_fit.predict(Xtest)
    
    plt.scatter(Xtest[:,1], ytest)
    plt.scatter(Xtest[:,1], ypredict)
    plt.show()