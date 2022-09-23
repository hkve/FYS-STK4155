'''
This is very messy, I have just done the first things that have come to mind, and I am rusty...
'''
from typing import OrderedDict
import numpy as np
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from linear_model import Model, LinearRegression
import resampling

class Data:
    '''
    Class for storing and handling (y, X) data
    '''
    def __init__(self, y, X, unscale=None):
        self.y = np.array(y)
        self.X = np.array(X)
        self.n_features = X.shape[-1]

        if unscale is None:
            self.unscale = lambda data : data
        elif callable(unscale):
            self.unscale = unscale
        else:
            raise TypeError("Specified unscaler is not callable.")

    def __len__(self):
        return self.n_features+1

    def __getitem__(self, key):
        try:
            i, j = key
            if i == 0:
                return self.y[j]
            elif i > 0 and i < self.X.shape[1]:
                return self.X[j, i]
            else:
                raise IndexError(f"Index {i} out of bounds for data of with {self.n_features} features.")
        except TypeError:
            if key==0:
                return self.y
            elif key > 0 and key < self.X.shape[1]:
                return self.X[:,key]
            else:
                raise IndexError(f"Index {key} out of bounds for data of with {self.n_features} features.")


    def __iter__(self):
        self.i = 0
        yield Data(np.array([self.y[self.i]]), np.array([self.X[self.i]]))

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


    def unpacked(self):
        '''
        Unpacks the Data into the 
        '''
        return self.y, self.X

    def sorted(self, axis=0):
        '''
        Returns sorted Data along specified axis. Index 0 refers to y, 1 etc. to features.
        '''
        sorted_idxs = np.argsort(self[axis])
        y_sorted = self.y[sorted_idxs]
        X_sorted = self.X[sorted_idxs,:]
        return Data(y_sorted, X_sorted, unscale=self.unscale)

    def shuffled(self):
        '''
        Returns shuffled Data.
        '''
        shuffled_idxs = np.arange(self.y.size); np.random.shuffle(shuffled_idxs)
        return Data(self.y[shuffled_idxs], self.X[shuffled_idxs], unscale=self.unscale)
    
    def unscaled(self):
        return self.unscale(self)

    def train_test_split(self, ratio=2/3, random_seed=None):
        '''
        Splits the data into training data and test data according to train_test-ratio
        '''
        np.random.seed(seed=random_seed) # allows control over random seed

        size = self.y.size
        split_idx = int(size*ratio)
        shuffled_idxs = np.arange(size); np.random.shuffle(shuffled_idxs)
        training_idxs = shuffled_idxs[:split_idx]
        test_idxs = shuffled_idxs[split_idx:]
        training_data = Data(
            self.y[training_idxs],
            self.X[training_idxs],
            unscale = self.unscale
        )
        test_data = Data(
            self.y[test_idxs],
            self.X[test_idxs],
            unscale = self.unscale
        )
        return training_data, test_data

    def mean(self):
        ''''
        Returns the mean of the y-data
        '''
        return np.mean(self.y)

    def var(self):
        '''
        Returns the variance of the y-data
        '''
        return np.mean((self.y-self.mean())**2)

    def scale(self, scheme="None"):
        return self.scalers_[scheme](self)

    def none_scaler_(data):
        '''
        Does not scale y, *X.
        '''
        return data

    def standard_scaler_(data):
        '''
        Scales y, *X to be N(0, 1)-distributed.
        '''
        # extracting data from Data-class to more versatile numpy array
        data = np.concatenate(([data.y], [*data.X.T])).T
        # vectorised scaling
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std = np.where(data_std != 0, data_std, np.inf) # sets unvaried data-columns to 0
        data_scaled = (data - data_mean) / data_std

        scaled_data = Data( # packing result into new Data-instance
            data_scaled[:,0],
            data_scaled[:,1:],
            unscale = lambda data : data*data_std + data_mean # teching new Data how to unscale
        )
        return scaled_data
    
    scalers_ = {
        "None" : none_scaler_,
        "Standard" : standard_scaler_
    }


# class TrainingFacility: # working title
#     def __init__(self, model, data):
#         '''
#         model : a Model class to undergo training etc.
#         data  : an instance of the Data class
#         '''
#         self.model = model
#         self.data = data

#         self.isFit = False
#         self.fitmodel = None
#         self.scaled_data = None

#     def fit_training_data(self, scaler="None", ratio=2/3, random_seed=None):
#         '''
#         Fits the model to training data.
#         NB. Not sure just how I want to implement this bit.
#         '''
#         scaled_data = self.data.scale(scheme=scaler)
#         self.training_data, self.test_data = scaled_data.train_test_split(ratio=ratio, random_seed=random_seed)
#         y_training, X_training = self.training_data.unpacked()
#         self.fit_model = self.model(method="pINV").fit(X_training, y_training)
#         self.isFit = True
#         return self.fit_model

#     def predict_test_data(self): # This is more proof of concept than useful method.
#         '''
#         Returns predicted Data after training.
#         '''
#         if self.isFit:
#             _, X_test = self.test_data.unpacked()
#             y_predict = self.fit_model.predict(X_test)
#             result = Data(y_predict, X_test, unscale=self.test_data.unscale)
#             return result
#         else:
#             raise Exception("Cannot make prediction, model has not yet been fitted to data.")

#     def diagnose_statistics(self):
#         '''
#         Returns a dictionary with the MSE and R2 of the models predicted data on the training and test data.
#         '''
#         y_train, X_train = self.training_data.unpacked()
#         y_test, X_test = self.test_data.unpacked()
#         y_train_predicted = self.fit_model.predict(X_train)
#         y_test_predicted = self.fit_model.predict(X_test)

#         statistics = {
#             "MSE_train" : self.fit_model.mse(y_train, y_train_predicted),
#             "R2_train" : self.fit_model.r2_score(y_train, y_train_predicted),
#             "MSE_test" : self.fit_model.mse(y_test, y_test_predicted),
#             "R2_test" : self.fit_model.r2_score(y_test, y_test_predicted)
#         }
#         return statistics


class Pipeline:
    def __init__(self, model:Model, data:Data, random_seed:int=None) -> None:
        '''
        model : a Model class to undergo training etc.
        data  : an instance of the Data class
        '''
        self.model = model
        self.data = data
        self.results = {}
        self.random_seed = random_seed

        self.fitting_data = data
        self.testing_data = data

    def __call__(self, steps:OrderedDict) -> dict:
        for step in steps.keys():
            assert steps in self.operations_.keys(), f"{step} is not a valid operation."
        for step, args in self.steps.items():
            self.operations_[step](**args)
        return self.results

    def scale_data_(self, method:str="None") -> None:
        self.data = self.data.scaled(method=method)
        self.fitting_data = self.fitting_data.scaled(method=method)
        self.testing_data = self.testing_data.scaled(method=method)


    def unscale_data_(self) -> None:
        self.data = self.data.unscaled()
        self.fitting_data = self.fitting_data.unscaled()
        self.testing_data = self.testing_data.unscaled()


    def train_test_split_(self, ratio:float=2/3) -> None:
        self.fitting_data, self.testing_data = self.data.train_test_split(ratio=ratio, random_seed=self.random_seed)

    def fit_model_(self) -> None:
        self.model.fit(self.fitting_data)

    def diagnose_statistics_(self, scoring:dict, resampler:str="None") -> None:
        resampler = self.resamplers_[resampler](
            data = self.fitting_data,
            reg = self.model,
            run = False,
            random_state = self.random_seed,
        )
        resampler.run(self.testing_data, scoring = scoring)
        for score in scoring:
            self.results[score] = resampler.scores_[score]


    resamplers_ = {
        "None" : resampling.NoneResampler,
        "Bootstrap" : resampling.Bootstrap,
        "Cross Validate" : resampling.CrossValidate,
    }

    operations_ = {
        "fit" : fit_model_,
        "scale" : scale_data_,
        "unscale" : unscale_data_,
        "split" : train_test_split_,
        "diagnose" : diagnose_statistics_,
    }



if __name__ == '__main__':
    # generating data to stuff down the pipe
    random_state = 321
    np.random.seed(random_state)

    x = np.random.uniform(0, 1, size=100)
    X = np.array([np.ones_like(x), x, x*x]).T
    y = np.exp(x*x) + 2*np.exp(-2*x) + 0.1*np.random.randn(x.size)
    data = Data(y, X) # storing the data and design matrix in Data-class


    # setting up Pipeline
    pipe = Pipeline(LinearRegression, data, random_seed=random_state)

    # example of steps
    steps = OrderedDict()
    steps["scale"] = {"method" : "Standard"}
    steps["split"] = {"ratio" : 3/4}
    steps["fit"] =  {}
    steps["unscale"] = {}
    steps["diagnose"] = {"scoring" : ("mse",), "resampler" : "None"}

    pipe(steps)


    '''
    # example of use

    import matplotlib.pyplot as plt

    # generating data to stuff down the pipe
    random_state = 321
    np.random.seed(random_state)
    x = np.random.uniform(0, 1, size=100)
    X = np.array([np.ones_like(x), x, x*x]).T
    y = np.exp(x*x) +2*np.exp(-2*x) + 0.1*np.random.randn(x.size)
    testdata = Data(y, X) # storing the data and design matrix in Data-class

    # builiding TrainingFacility with the LinearRegression-class from linear_model.py
    tester = TrainingFacility(LinearRegression, testdata)

    # fitting the model
    fitted_linear_model = tester.fit_training_data(
        scaler = "None",
        ratio = 3/4,
        random_seed = random_state
    )
    # predicting the test data
    predicted_data = tester.predict_test_data().unscaled()
    # this returns an instance of Data-class

    # Generating useful statistics
    for statistic, value in tester.diagnose_statistics().items():
        print(statistic + f" score is {value:.5f}")

    # showing off some functionality of Data-class
    sorted_data = predicted_data.sorted(axis=2) # sorts by x-values
    ysorted, Xsorted = sorted_data.unpacked()

    # simple (simple...) plot
    ytest, Xtest = tester.test_data.unscaled().unpacked()
    plt.scatter(Xtest[:,1], ytest)
    plt.plot(Xsorted[:,1], ysorted)

    plt.show()
    '''