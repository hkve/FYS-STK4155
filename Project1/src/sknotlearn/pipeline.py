'''
This is very messy, I have just done the first things that have come to mind, and I am rusty...
'''
from typing import OrderedDict
import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import Data
from linear_model import Model, LinearRegression
import resampling


class Pipeline:
    
    def __init__(self, model:Model, data:Data, random_state:int=None) -> None:
        """_summary_

        Args:
            model (Model): _description_
            data (Data): _description_
            random_state (int, optional): _description_. Defaults to None.
        """
        self.model = model
        self.fitted_model = None
        self.data = data
        self.results = {}
        self.random_state = random_state

        self.fitting_data = data
        self.testing_data = data

    def do(self, steps:OrderedDict) -> dict:
        """_summary_

        Args:
            steps (OrderedDict): _description_

        Returns:
            dict: _description_
        """
        for operation in steps.keys():
            assert operation in self.operations_.keys(), f"{operation} is not a valid operation."
        for operation, kwargs in steps.items():
            self.operations_[operation](self, **kwargs)
        return self.results

    def scale_data_(self, scheme:str="None") -> None:
        """_summary_

        Args:
            scheme (str, optional): _description_. Defaults to "None".
        """
        self.data = self.data.scaled(scheme=scheme)
        self.fitting_data = self.fitting_data.scaled(scheme=scheme)
        self.testing_data = self.testing_data.scaled(scheme=scheme)

    def unscale_data_(self) -> None:
        """_summary_
        """
        self.data = self.data.unscaled()
        self.fitting_data = self.fitting_data.unscaled()
        self.testing_data = self.testing_data.unscaled()


    def train_test_split_(self, ratio:float=2/3) -> None:
        """_summary_

        Args:
            ratio (float, optional): _description_. Defaults to 2/3.
        """
        self.fitting_data, self.testing_data = self.data.train_test_split(ratio=ratio, random_state=self.random_state)

    def initialize_model_(self, reg_args:dict) -> None:
        """_summary_

        Args:
            reg_args (dict): Keyword arguments to pass on to model.__init__.
        """
        self.fitted_model = self.model(**reg_args)

    def diagnose_statistics_(self, scoring:tuple, resampler:str="None") -> None:
        """_summary_

        Args:
            scoring (tuple): _description_
            resampler (str, optional): _description_. Defaults to "None".
        """
        resampler = self.resamplers_[resampler](
            data = self.fitting_data,
            reg = self.fitted_model,
            run = False,
            random_state = self.random_state,
        )
        resampler.run(self.testing_data, scoring=scoring)
        for score in scoring:
            self.results[score] = resampler.scores_[score]

    def fit_parameter_(self, param_name, param_vals, resampler="None") -> None:
        """_summary_

        Args:
            params (dict): _description_
        """
        original_results = self.results

        best_val = param_vals[0]
        best_mse = np.inf
        for val in param_vals[1:]:
            self.initialize_model_(param_name = val)
            self.diagnose_statistics_(scoring=("mse"), resampler=resampler)
            if np.mean(self.results["mse"]) < best_mse:
                best_val = val
        self.initialize_model_(param_name = best_val)
        self.results[param_name] = best_val

        self.results = original_results

    def print_results(self) -> None:
        """Prints the results that have been gathered"""
        for score, result in self.results:
            print(score + f" : {result}")

    resamplers_ = {
        "None" : resampling.NoneResampler,
        "Bootstrap" : resampling.Bootstrap,
        "Cross Validate" : resampling.KFold_cross_validate,
    }

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
    pipe = Pipeline(LinearRegression, data, random_state=random_state)

    # example of steps
    steps = OrderedDict()
    steps["scale"] = {"scheme" : "Standard"}
    steps["split"] = {"ratio" : 3/4}
    # steps["init"] =  {}
    steps["unscale"] = {}
    # steps["diagnose"] = {"scoring" : ("mse",), "resampler" : "None"}

    pipe.do(steps)