import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA




def randomGuesses(
            num_matches : int, 
            odds : list = [1/3, 1/3, 1/3]) -> np.ndarray:
    """
    Generate a set of random guesses for the result (W, D, L) for comparison.

    Parameters
    ----------
    num_matches : int
        number of matches in the set for which to compare with
    odds : list, optional
        naive distribution of wins, draws, losses, respectively, by default [1/3, 1/3, 1/3]

    Returns
    -------
    np.ndarray
        result matrix
    """
    result = np.zeros((num_matches, 3))
    # iterate through matches (as I did not manage to do it directly...)
    for m in range(num_matches):
        # multinomial?
        x = np.random.choice([0,1,2], p=odds)
        result[m, x] = 1

    # maybe make dataframe and map??
    return result




if __name__ == "__main__":

    print("\n >>> \n")

    import context
    from sknotlearn.datasets import load_EPL, get_result_distribution

    container = load_EPL(True)
    
    trainx, testx, trainy, testy = train_test_split(container.x, container.y)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(trainx)
    num_matches = len(trainx)

    guessed_results = randomGuesses(num_matches, get_result_distribution())    

    pca = PCA()
    principal_components = pca.fit_transform(scaled_data)
    principal_data = pd.DataFrame(data=principal_components)
    cum_var_ratio = np.cumsum(pca.explained_variance_ratio_)

    final_data = pd.concat([principal_data, trainy.reset_index(drop=True)], axis=1)
    print(final_data.head())

    thresh = 0.99
    n_features = list(cum_var_ratio).index(cum_var_ratio[cum_var_ratio >= thresh][0])
    print(f"Best #features = {n_features}")

    fig, ax = plt.subplots()
    ax.plot(cum_var_ratio)
    ax.axvline(n_features, ls='--')
    
    ax.set_xlabel(r"\# PCA features")
    ax.set_ylabel(r"Variance [\%]")

    plt.show()
   

    print("\n --- \n")


