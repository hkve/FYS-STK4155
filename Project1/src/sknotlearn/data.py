import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class Data:
    """Class for storing data y with corresponding design matrix X in a (y, X) type object.
    Indexing Data object is done as
        data[i] = Data(y[i], X[i]), giving the i'th data point
        data[i,j] = y[i] or X[i,j] where j=0 refers to y, and j=1,... refers to the features.

    Public methods:
    Data is unpacked to its np.ndarrays by
        y, X = data.unpacked()
    Data can be scaled by
        data_scaled = data.scaled(method)
    Data can be scaled back by
        data = data_scaled.unscaled()
    Data can shuffled by
        data = data.shuffled(with_replacement=True/False)
    Data can be sorted by feature by
        data = data.sorted(axis)
    Data can be split into train and test Data by
        data_train, data_test = data.train_test_split(ratio)
    Features can be expanded to polynomials by
        data = data.polynomial(degree)"""
    def __init__(self, y:np.ndarray, X:np.ndarray, unscale=None) -> None:
        self.y = np.array(y)
        self.X = np.array(X)
        self.n_features = X.shape[-1]

        # initiating unscale function, defaults to trivial function
        self.unscale = unscale or (lambda data : data)
        assert callable(self.unscale), f"Specified unscaler is not callable (is {type(self.unscaler)})"

    # The following methods are there for indexing and iteration of Data
    def __len__(self) -> int:
        return self.n_features+1

    def __getitem__(self, key):
        try:
            i, j = key
            if j == 0:
                return self.y[i]
            elif j > 0 and j < self.X.shape[1]:
                return self.X[i, j]
            else:
                raise IndexError(f"Index {j} out of bounds for Data with {self.n_features} features.")
        except:
            return Data(self.y[key], self.X[key])

    def __iter__(self):
        self.i = 0
        yield Data(np.array([self.y[self.i]]), np.array([self.X[self.i]]))

    def __next__(self):
        self.i += 1

    # Printing for debugging.
    def __str__(self):
        return str(np.concatenate(([self.y], [*self.X.T])).T)

    # The following methods are there for arithmetics.
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

    # The remaining methods do stuff
    def unpacked(self): #-> tuple[np.ndarray, np.ndarray]:

        """Unpacks the Data into the y and X ndarrays

        Returns:
            tuple[np.ndarray, np.ndarray]: _description_
        """
        return self.y, self.X

    def sorted(self, axis:int=0):
        """Sorts Data along specified axis.

        Args:
            axis (int, optional): Feature to sort against. 0 is y, 1 etc. are features. Defaults to 0.

        Returns:
            Data: sorted Data
        """
        sorted_idxs = np.argsort(self[:,axis])
        y_sorted = self.y[sorted_idxs]
        X_sorted = self.X[sorted_idxs,:]
        return Data(y_sorted, X_sorted, unscale=self.unscale)

    def shuffled(self, with_replacement:bool=False, random_state:int=None):
        """Returns shuffled Data.

        Args:
            with_replacement (bool, optional): _description_. Defaults to False.
            random_state (int, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        np.random.seed(random_state)
        if with_replacement:
            shuffled_idxs = np.random.randint(0, self.y.size, self.y.size)
        else:
            shuffled_idxs = np.arange(self.y.size); np.random.shuffle(shuffled_idxs)
        return Data(self.y[shuffled_idxs], self.X[shuffled_idxs], unscale=self.unscale)
    
    def unscaled(self):
        """Returns unscaled Data

        Returns:
            _type_: _description_
        """
        return self.unscale(self)

    def train_test_split(self, ratio:float=2/3, shuffle:bool=True, random_state:int=None) -> tuple:
        """Splits the data into training data and test data according to train_test-ratio.

        Args:
            ratio (float, optional): _description_. Defaults to 2/3.
            shuffle (bool, optional): Whether to shuffle data before splitting. Defaults to True
            random_state (int, optional): _description_. Defaults to None.

        Returns:
            tuple: _description_
        """
        size = self.y.size
        idxs = np.arange(size)
        split_idx = int(size*ratio)
        if shuffle:
            np.random.seed(random_state) # allows control over random seed
            np.random.shuffle(idxs)

        training_idxs = idxs[:split_idx]
        test_idxs = idxs[split_idx:]
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

    def mean(self) -> float:
        """Returns the mean of the y-data.

        Returns:
            float: _description_
        """
        return np.mean(self.y)

    def var(self) -> float:
        """Returns the variance of the y-data.

        Returns:
            float: _description_
        """
        return np.mean((self.y-self.mean())**2)

    def scaled(self, scheme:str="None"):
        """Returns scaled Data.

        Args:
            scheme (str, optional): _description_. Defaults to "None".

        Returns:
            Data: _description_
        """
        return self.scalers_[scheme](self)

    def none_scaler_(data):
        """Does not scale y, *X.

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        return data

    def polynomial(self, degree:int, return_powers:bool=False):
        """
        Makes polynomials based on features (columns) in X. Optionally,
        the ordering of powers can be saved to data object.

        Args:
            degree (int): Highest degree to include in polynomial
            save_powers (bool): If the ordering of power should be saved. Defaults to False

        Returns:
            Data: at Data-instance with a new design matrix based on polynomials of prior features.
        """
        poly = PolynomialFeatures(degree=degree)
        X = poly.fit_transform(self.X)
        out = Data(self.y, X)
        
        if return_powers:
            return out, poly.powers_
        else:
            return out

    def standard_scaler_(data):
        """Scales y, *X to be N(0, 1)-distributed.

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        # extracting data from Data-class to more versatile numpy array
        data = np.concatenate(([data.y], [*data.X.T])).T
        # vectorised scaling
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std = np.where(data_std != 0, data_std, 1) # sets unvaried data-columns to 0
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