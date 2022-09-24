from typing import OrderedDict
import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import Data
from linear_model import Model, LinearRegression, Ridge, Lasso
import resampling


class Pipeline:
    """
    Pipeline for exploring a linear_moder.Model against data. More comprehensive description will come.
    """
    def __init__(self, model:Model, data:Data, random_state:int=None) -> None:
        """Constructor.

        Args:
            model (Model): A class (not instance) of linear_model to be used.
            data (Data): A Data-class instance containing targets y with corresponding design matrix X. To be used for training and/or testing.
            random_state (int, optional): np.random seed for control. Defaults to None.
        """
        self.model = model
        self.fitted_model = None # instance of model to be made and fitted later
        self.data = data
        self.results = {} # dict of diagnostic results.
        self.random_state = random_state

        self.fitting_data = data # data used for training
        self.testing_data = data # data used for diagnostics

    def do(self, steps:OrderedDict) -> dict:
        """Executes steps in order.

        Args:
            steps (OrderedDict): An ordered dictionary where the keys denote the operations and the entries are dictionaries containing args to be passed to the operation.

        Returns:
            dict: Contains any results that have been calculated during the steps.
        """
        # First asserting that all the operations are valid
        for operation in steps.keys():
            assert operation in self.operations_.keys(), f"{operation} is not a valid operation."
        # Performing the operations
        for operation, kwargs in steps.items():
            self.operations_[operation](self, **kwargs)
        return self.results

    def scale_data_(self, scheme:str="None") -> None:
        """Scales all data in the pipeline according to scheme.

        Args:
            scheme (str, optional): Scaling scheme. "None" and "Standard" available. Defaults to "None".
        """
        self.data = self.data.scaled(scheme=scheme)
        self.fitting_data = self.fitting_data.scaled(scheme=scheme)
        self.testing_data = self.testing_data.scaled(scheme=scheme)

    def unscale_data_(self) -> None:
        """Unscales all data in the pipeline. If data is not scaled, this does nothing.
        """
        self.data = self.data.unscaled()
        self.fitting_data = self.fitting_data.unscaled()
        self.testing_data = self.testing_data.unscaled()


    def train_test_split_(self, ratio:float=2/3) -> None:
        """Splits the inital data of the pipeline into one part which is used for fitting the regression Model and one part which is used for testing and extracting diagnostics.

        Args:
            ratio (float, optional): Ratio of training data to test data. Defaults to 2/3.
        """
        self.fitting_data, self.testing_data = self.data.train_test_split(ratio=ratio, random_state=self.random_state)

    def initialize_model_(self, reg_args:dict) -> None:
        """Sets up an instance of the pipeline regression Model, with specified arguments passed, to be used for fitting and diagnostics.

        Args:
            reg_args (dict): Keyword arguments to pass on to Model constructor.
        """
        self.fitted_model = self.model(**reg_args)

    def diagnose_statistics_(self, scoring:tuple[str], resampler:str="None") -> None:
        """Uses specified resampling method to fit pre-initialised Model to fitting_data and extract specified statistical metrics from predictions made on testing_data.

        Args:
            scoring (tuple[str]): String names of score to be evaluated
            resampler (str, optional): String name of resampler to use for extracting statistics. "None", "Bootstrap" and "Cross Validate" available. Defaults to "None".
        """
        # Creating instance of resampler
        resampler = self.resamplers_[resampler](
            data = self.fitting_data,
            reg = self.fitted_model,
            run = False,
            random_state = self.random_state,
        )
        resampler.run(self.testing_data, scoring=scoring)
        # save scores from resampled run
        for score in scoring:
            self.results[score] = resampler.scores_[score]

    def fit_parameter_(self, param_name:str, param_vals:np.ndarray, reg_args:dict, resampler="None") -> None:
        """Grid search param_name of self.model over param_vals to find the value which which results in lowest MSE on testing_data.

        Args:
            param_name (str): Keyword of parameter to pass to Model.
            param_vals (Iterable): Iterable of values for given parameter to compare.
            reg_args (dict): Other kwargs to pass to Model
            resampler (str, optional): Resampling method to employ when calculating MSEs. Defaults to "None".
        """

        original_results = self.results # saving results already made, as they will temporarily be overwritten

        # initializing search values
        best_val = None
        best_mse = np.inf
        for val in param_vals:
            reg_args[param_name] = val # adding search parameter to args to pass to self.model.
            self.initialize_model_(reg_args = reg_args) # setting up instance of self.model.
            self.diagnose_statistics_(scoring=("mse",), resampler=resampler) # calculating MSE
            # when resampling, all MSEs are saved, so comparing mean values. (This could be changed.)
            mse = np.mean(self.results["mse"])
            # Check for improvement
            if mse < best_mse:
                best_val = val
                best_mse = mse
        # finally setting self.fitted_model to be an instance with the best parameter value.
        reg_args[param_name] = best_val
        self.initialize_model_(reg_args=reg_args)
        # saving best parameter value to results
        self.results[param_name] = best_val

        self.results = original_results

    def print_results(self) -> None:
        """Prints the results that have been gathered."""
        for score, result in self.results.items():
            print(score + f" : {result}")

    # Dictionary with available resamplers
    resamplers_ = {
        "None" : resampling.NoneResampler,
        "Bootstrap" : resampling.Bootstrap,
        "Cross Validate" : resampling.KFold_cross_validate,
    }

    # Dictionary with available operations
    operations_ = {
        "init" : initialize_model_,
        "scale" : scale_data_,
        "unscale" : unscale_data_,
        "split" : train_test_split_,
        "diagnose" : diagnose_statistics_,
        "tune" : fit_parameter_,
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
    pipe = Pipeline(Ridge, data, random_state=random_state)

    # example of steps
    steps = OrderedDict()
    steps["scale"] = {"scheme" : "Standard"}
    steps["split"] = {"ratio" : 3/4}
    # steps["init"] =  {"reg_args" : {"lmbda" : 1e-3, "method" : "INV"}}
    steps["tune"] = {
        "param_name" : "lmbda",
        "param_vals" : np.logspace(-12,-6,11),
        "reg_args" : {"method" : "INV"}
    }
    steps["unscale"] = {}
    steps["diagnose"] = {"scoring" : ("mse", "bias2", "var"), "resampler" : "None"}

    # performing steps and printing results
    pipe.do(steps)
    pipe.print_results()
