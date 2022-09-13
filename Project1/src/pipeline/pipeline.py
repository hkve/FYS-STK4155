'''
This is very messy, I have just done the first things that have come to mind, and I am rusty...
'''
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

        self.scalers_ = {
            "Standard" : self.standard_scaler_
        }

        self.unscale = lambda self : self

    def __getitem__(self, key):
        i, j = key
        if i == 0:
            return self.y[j]
        elif i > 0 and i < X.shape[1]:
            return X[j, i]
        else:
            raise IndexError(f"Index i out of bounds for data of with {self.n_features} features.")

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

    def train_test_split(self, ratio=2/3):
        '''
        Splits the data into training data and test data according to train_test-ratio
        '''
        size = data.shape[0]
        np.random.shuffle(data)
        training_data, test_data = data[:int(size*ratio)], data[int(size*ratio):]
        return training_data, test_data

    def scale(self, scheme="Standard"):
        return self.scalers_[scheme](self)

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


class TrainingFacility: # working title
    def __init__(self, model, data):
        '''
        model : a Model class to undergo training etc.
        data  : array of data in shape (y, *X), where *X denotes columns of X.
        '''
        self.model = model
        self.data = data

        self.scaled_data = None
        self.unscale_data = lambda scaled_data : None

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
        return np.c_[y, X]

    def scale_data(self, scheme="Standard"):
        '''
        Scales the data according to scheme

        NB. Schemes not implemented yet, deafults to Standard scaling of y, x to be N(1,0)-distributed.
        '''
        self.scaled_data = self.scalers_[scheme](self.data)

        return self

    def train_test_split(self, data, train_test=2/3):
        '''
        Splits the data into training data and test data according to train_test-ratio
        '''
        size = data.shape[0]
        np.random.shuffle(data)
        training_data, test_data = data[:int(size*train_test)], data[int(size*train_test):]
        return training_data, test_data

    def fit_training_data(self, scale=False, train_test=2/3):
        '''
        Fits the model to training data.
        NB. Not sure just how I want to implement this bit.
        '''
        if scale:
            self.scale_data()
            data = self.scaled_data
            # print(self.scaled_data[:,:1].shape, self.scaled_data[:,2:].shape)
            # sys.exit()
            data = np.concatenate((self.scaled_data[:,:1], self.scaled_data[:,2:]), axis=1)
        else:
            data = self.data
        training_data, self.test_data = self.train_test_split(data, train_test=train_test)
        y_training, X_training = self.unpack_data_(training_data)
        fit_model = self.model(method="INV").fit(X_training, y_training)
        return fit_model

    def standard_scaler_(self, data):
        '''
        Scales y, *X to be N(0, 1)-distributed.
        '''
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std = np.where(data_std != 0, data_std, 1) # sets unvaried data-columns to 0
        data_scaled = (data - data_mean) / data_std

        self.unscale_data = lambda scaled_data : scaled_data*data_std + data_mean

        return data_scaled


if __name__ == '__main__':
    x = np.random.uniform(0, 1, size=10)
    X = np.array([np.ones_like(x), x, x**2]).T
    y = np.exp(x*x) +2*np.exp(-2*x) + 0.1*np.random.randn(x.size)
    testdata = Data(y, X)
    testdata_scaled = testdata.scale()


    # testing with random function I thought of
    # setting up first
    from sknotlearn.linear_model import LinearRegression
    x = np.random.uniform(0, 1, size=1000)
    X = np.array([np.ones_like(x), x, x**2]).T
    beta = np.array([1, -2, 3])
    y = np.exp(x*x) + 2*np.exp(-2*x) + 0.1*np.random.randn(x.size)
    # y = X@beta + 0.1*np.random.randn(x.size)
    data = np.array([(yi, 1, xi, xi**2) for yi, xi in zip(y,x)])

    # testing some functionality
    test = TrainingFacility(LinearRegression, data)
    fit = test.fit_training_data(scale=True)
    ypredict = test.unscale_data(fit.predict(test.test_data[:,2:]))

    import matplotlib.pyplot as plt
    plt.scatter(test.test_data[:,2], test.test_data[:,0])
    plt.scatter(test.test_data[:,2], ypredict)
