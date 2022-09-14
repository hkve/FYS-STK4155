from re import X
import numpy as np

class cross_validate:
    def __init__(self, reg, y, X, k=5, scoring=("mse"), return_train_score=False):
        self.params_ = {}

        if type(scoring) not in [list, tuple]:
            scoring = (scoring, )

        self.run(reg, X, y, k, scoring)

    def __str__(self):
        return str(self.params_)

    def run(self, reg, X, y, k, scoring):
        np.random.shuffle(X)
        


if __name__ == "__main__":
    n = 1000
    x = np.random.uniform(-1, 1, n)
    y = 1 + 2*x

    X = np.c_[np.ones_like(x), x]

    from linear_model import LinearRegression

    reg = LinearRegression()
    cv = cross_validate(reg, y, X, k=4, scoring="mse", return_train_score=True)