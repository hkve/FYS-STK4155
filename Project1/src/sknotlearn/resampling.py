import numpy as np

# class Bootstrap: 
#     def __init__(self, reg,):


class cross_validate:
    """
    Class to preform k-fold cross validation given a regression instance derived from
    sknotlearn.linear_model.model. Taking X as your designe matrix and y as your target values, preform
    fitting and computation of metrics based on scoring for each of the k folds. Optionally,
    an initial shuffeling of the data can be preform and random_state can be set for 
    reproducibility.
    """
    def __init__(self, reg, y, X, k=5, scoring=("mse"), shuffle=True, run=True, random_state=321):
        """
        Constructor of class. Sets up scoring dict, set seeds and optionally runs cv. 

        Args:
            reg: (Instance derived from sknotlearn.linear_model.model) Instance of regression class used for cv
            y: (np.array) Target values used for cross validation
            X: (np.array) Design matrix corresponding to y
            k: (int) How many folds should be used. Defaults to 5
            scoring: (str/iterable(str)) What scoring methods should be used for evaluation
            shuffle: (bool) If X and y should be shuffled before creating folds
            run: (bool) If cv should run after calling constructor.
            random_state: (int) Seed to use, only relevant if shuffel = True
        """

        # If scoring is not list/tuple, make it a tuple
        if type(scoring) not in [list, tuple]:
            scoring = (scoring, )
        
        # Set seed
        np.random.seed(random_state)

        # Compute for both train and test
        cases = ["train", "test"]

        # Setup scoring dict
        self.scores_ = {}
        for case in cases:
            for score in scoring:
                assert score in reg.metrics_.keys(), f"The score {score} is not avalible in model {type(reg)}"
                self.scores_[f"{case}_{score}"] = []

        self.shuffle_ = shuffle
        self.scoring_ = scoring
        self.k_ = k

        if run: self.run(reg, X, y)


    def __str__(self):
        """
        Eye pleaser for viewing computed scores. 

        Returns:
            to_return: (str) Formatted string containing scores for each fold 
        """

        # Make title
        scoring = list(self.scores_.keys())
        k = len(self.scores_[scoring[0]])
        to_return = f"Cross validation with k = {k}. Shuffel of input data = {self.shuffle_}\n"
        
        # Format values
        for score in scoring:
            to_return += f"{score:<20}"
        to_return += "\n"

        for i in range(k):
            for score in scoring:
                to_return += f"{self.scores_[score][i]:<20.4f}"

            to_return += "\n"

        return to_return


    # Implement simple dict functionality, should only be used for extracting things from scores_
    def __getitem__(self, key):
        return self.scores_[key]

    def keys(self):
        return self.scores_.keys()

    def values(self):
        return self.scores_.values()

    def items(self):
        return self.scores_.items()


    def divide_work_(self, n, k):
        """
        Makes start and stop indicies given n data points across k folds. This
        is returned in two lists of lenght k.

        Args:
            n: (int) Number of datapoints in X/y
            k: (int) Number of folds to create 
        
        Returns:
            start_fold: (list[int]) Contains the index of where each fold should start
            end_fold: (list[int]) Contains the index of where each fold should end
        """

        # Int div to divide equally, and calculate rest case where n%k != 0
        n_each_fold = int(n/k)
        n_extra = n%k

        # Setup first start and first stop
        start_fold = [0]*k
        stop_fold = [0]*k
        stop_fold[0] = n_each_fold + (n_extra != 0)

        # Divde the rest of the work. The +1 folds in n%k != 0 case is given to the first n%k folds.
        for i in range(1, k):
            start_fold[i] = stop_fold[i-1]
            stop_fold[i] = start_fold[i] + n_each_fold
            
            if i < n_extra: stop_fold[i] += 1

        return start_fold, stop_fold


    def run(self, reg, X, y):
        """
        Main run function. Given reg, X and y, preform the k-fold cross validation algorithm.

        Args:
            reg: (Instance derived from sknotlearn.linear_model.model) Instance used for fitting
            X: (np.array) Design matrix corresponding to y
            y: (np.array) Target values used for cross validation
        """

        k, scoring = self.k_, self.scoring_

        n, _ = X.shape

        # Make array containg all indicies
        idx = np.arange(n)

        # If they should be shuffled, shuffle them
        if self.shuffle_:
            np.random.shuffle(idx)
            X = X[idx, :]
            y = y[idx]
            idx = np.arange(n) # Redefine since idx is now shuffled

        # Get indicies of partitioned folds
        start_fold, stop_fold = self.divide_work_(n, k)
                
        # Iterate over folds, i referes to the holdout set
        for i in range(k):
            # Define indicies holdout set
            test_idx = np.arange(start_fold[i], stop_fold[i])
            
            # Define indicies of train. Takes everything from 0 to holdout and form end of holdout to n
            train_idx = idx[:start_fold[i]]
            train_idx = np.r_[train_idx, idx[stop_fold[i]:]]
        
            # Create (X,y) from train and holdout set
            X_test, y_test = X[test_idx,:], y[test_idx]
            X_train, y_train = X[train_idx,:], y[train_idx]

            reg.fit(X_train, y_train)

            y_pred_train = reg.predict(X_train)
            y_pred_test = reg.predict(X_test)

            # Compute the scores for train and test            
            for score in scoring:
                score_train = reg.metrics_[score](y_train, y_pred_train)
                score_test = reg.metrics_[score](y_test, y_pred_test)
                
                self.scores_[f"test_{score}"].append(score_test)
                self.scores_[f"train_{score}"].append(score_train)

        # Make scores np.arrays such that math is easy
        for k, v in self.items():
            self.scores_[k] = np.array(v)


if __name__ == "__main__":
    n = 1000
    x = np.random.uniform(-1, 1, n)
    y = 1 + x**2 + 0.2*np.random.normal(loc=0, scale=1, size=n)


    X = np.c_[np.ones_like(x), x, x**2]

    from linear_model import LinearRegression

    reg = LinearRegression()
    cv = cross_validate(reg, y, X, k=4, scoring=("mse", "r2_score"))
    print(cv["train_mse"].mean(), cv["train_mse"])