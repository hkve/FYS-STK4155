from random import shuffle
import numpy as np

class cross_validate:
    def __init__(self, reg, y, X, k=5, scoring=("mse"), shuffle=True, random_state=321):
        if type(scoring) not in [list, tuple]:
            scoring = (scoring, )
        
        np.random.seed(random_state)
        cases = ["train", "test"]
        self.params_ = {}

        for case in cases:
            for score in scoring:
                assert score in reg.metrics_.keys(), f"The score {score} is not avalible in model {type(reg)}"
                self.params_[f"{case}_{score}"] = []

        self.shuffle_ = shuffle
        self.scoring_ = scoring
        self.k_ = k

        self.run(reg, X, y)


    def __str__(self):
        scoring = list(self.params_.keys())
        k = len(self.params_[scoring[0]])
        to_return = f"Cross validation with k = {k}. Shuffel of input data = {self.shuffle_}\n"
        for score in scoring:
            to_return += f"{score:<20}"
        to_return += "\n"

        for i in range(k):
            for score in scoring:
                to_return += f"{self.params_[score][i]:<20.4f}"

            to_return += "\n"
        return to_return


    def divide_work_(self, n, k):
        n_each_fold = int(n/k)
        n_extra = n%k

        start_fold = [0]*k
        stop_fold = [0]*k
        stop_fold[0] = n_each_fold + (n_extra != 0)

        for i in range(1, k):
            start_fold[i] = stop_fold[i-1]
            stop_fold[i] = start_fold[i] + n_each_fold
            if i < n_extra:
                stop_fold[i] += 1

        return start_fold, stop_fold


    def run(self, reg, X, y):
        k, scoring = self.k_, self.scoring_
        
        n, _ = X.shape
        idx = np.arange(n)

        if self.shuffle_:
            np.random.shuffle(idx)
            X = X[idx, :]
            y = y[idx]
            idx = np.arange(n)

        start_fold, stop_fold = self.divide_work_(n, k)
                
        for i in range(k):
            test_idx = np.arange(start_fold[i], stop_fold[i])
            train_idx = idx[:start_fold[i]]
            train_idx = np.r_[train_idx, idx[stop_fold[i]:]]
        
            X_test, y_test = X[test_idx,:], y[test_idx]
            X_train, y_train = X[train_idx,:], y[train_idx]

            reg.fit(X_train, y_train)

            y_pred_train = reg.predict(X_train)
            y_pred_test = reg.predict(X_test)
            
            for score in scoring:
                score_train = reg.metrics_[score](y_train, y_pred_train)
                score_test = reg.metrics_[score](y_test, y_pred_test)
                
                self.params_[f"test_{score}"].append(score_test)
                self.params_[f"train_{score}"].append(score_train)

if __name__ == "__main__":
    
    n = 1000
    x = np.random.uniform(-1, 1, n)
    y = 1 + x**2 + 0.2*np.random.normal(loc=0, scale=1, size=n)


    X = np.c_[np.ones_like(x), x, x**2]

    from linear_model import LinearRegression

    reg = LinearRegression()
    cv = cross_validate(reg, y, X, k=4, scoring=("mse", "r2_score"))
    print(cv)