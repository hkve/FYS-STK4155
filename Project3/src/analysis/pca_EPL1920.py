import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import plot_utils
from collections import namedtuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


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
    np.random.seed(12345)
    result = np.zeros(num_matches)
    # iterate through matches (as I did not manage to do it directly...)
    for m in range(num_matches):
        # multinomial?
        x = np.random.choice([2,1,0], p=odds)
        result[m] = x

    # maybe make dataframe and map??
    return result


def get_random_accuarcy(
            data : pd.DataFrame,
            distribution : str = "estimated") -> tuple:
    """
    Get the accuracy for a random guess, given that it knows the EPL trend:
        * 50 % chance of home win
        * 30 % chance of away win
        * 20 % chance of draw

    Parameters
    ----------
    data : pd.DataFrame
        the non-endoded data from load_EPL()
    distribution : str, optional
        distribution for random sampling, by default "estimated"
            "estimated" - 50 % chance of home win, 30 % chance of away win, 20 % chance of draw
            "uniform" - 1/3 chance of home win, 1/3 chance of away win, 1/3 chance of draw
            "home wins" - 100 % chance of home win, 0 % chance of away win, 0 % chance of draw
        
    Returns
    -------
    float
        accuracy
    """
    home_result = data.loc[data["ground"] == "h"]["result"].reset_index(drop=True, inplace=False)
    away_result = data.loc[data["ground"] == "a"]["result"].reset_index(drop=True, inplace=False)
    
    if distribution == "estimated":
        ### 50 % chance of home win, 30 % chance of away win, 20 % chance of draw
        guess_home = randomGuesses(len(home_result), [0.5,0.2,0.3])
        guess_away = randomGuesses(len(away_result), [0.3,0.2,0.5])
    elif distribution == "uniform":
        ### 1/3 chance of home win, 1/3 chance of away win, 1/3 chance of draw
        guess_home = randomGuesses(len(home_result), [1/3,1/3,1/3])
        guess_away = randomGuesses(len(away_result), [1/3,1/3,1/3])
    elif distribution == "home wins":
        ### 100 % chance of home win, 0 % chance of away win, 0 % chance of draw
        guess_home = randomGuesses(len(home_result), [1,0,0])
        guess_away = randomGuesses(len(away_result), [0,0,1])
    else:
        print("Provide valid argument")
        

    
    res_map = {'w': 2, 'd': 1, 'l': 0}
    home_result["actual"] = home_result.map(res_map)
    away_result["actual"] = away_result.map(res_map)
 
    ## brute force (why does it not work otherwise???):
    actual_home = np.zeros_like(guess_home)
    actual_away = np.zeros_like(guess_away)

    for j in range(len(actual_away)):
        actual_home[j] = home_result["actual"][j]
        actual_away[j] = away_result["actual"][j]
    
    
    actual = np.append(actual_home, actual_away)
    guess = np.append(guess_home, guess_away)

    accuracy = accuracy_score(actual, guess)

    return accuracy


def generate_explained_variance_plot(
            trainx: pd.DataFrame,
            SHOW: bool = False) -> None:
    """Function to generate and plot the explained variance from principal component analysis. It is plotted both as a step-wise cumulative function and with histogram bars showing the explained variance of each principal component.

    Args:
        trainx (pd.DataFrame): Pandas data frame with the unscaled data in the basis of the original features.
        SHOW (bool): If true, the plot is also shown. Default is False.

    """
    scaler = StandardScaler()
    trainx = scaler.fit_transform(trainx)

    num_matches = len(trainx)
    n_features_org = np.shape(trainx)[1]

    pca = PCA()
    principal_components = pca.fit_transform(trainx)

    exp_var = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(exp_var)
    thresh = 0.99
    n_features = list(cum_var_ratio).index(cum_var_ratio[cum_var_ratio >= thresh][0])
    # print(f"Best #features = {n_features}")

    # print("\n", np.shape(principal_components), "\n..")

    principal_data = pd.DataFrame(data=principal_components[:,:n_features])
    
    final_data = pd.concat([principal_data, trainy.reset_index(drop=True)], axis=1)
    print(final_data.head())
    print(f"Total features: {trainx.shape[1]}")


    fig, ax = plt.subplots()
    ax.bar(range(1,n_features_org+1), exp_var, alpha=0.5)
    ax.step(range(1,n_features_org+1),cum_var_ratio)
    ax.axvline(n_features, ls='--', label=r"PCs: {ps:.0f}".format(ps=n_features))
    
    ax.set_xlabel(r"Principal components")
    ax.set_ylabel(r"Explained variance")
    ax.legend()

    plot_utils.save("pca_pl")
    plt.show()


if __name__ == "__main__":

    print("\n >>> \n")

    import context
    from sknotlearn.datasets import load_EPL, get_result_distribution

    ### Get encoded data
    container = load_EPL(True)
    trainx, testx, trainy, testy = train_test_split(container.x, container.y, test_size=1/6)
    n_train = len(trainx)
    n_test = len(testx)


    ### Guess randomly (on test data)
    data0 = load_EPL(False, False).iloc[n_train:]
    ref_accuracy_smart  = get_random_accuarcy(data0)
    ref_accuracy_octopus  = get_random_accuarcy(data0, "uniform")
    ref_accuracy_baby  = get_random_accuarcy(data0, "home wins")
    print(f"Accuracy from learned random guesser: {ref_accuracy_smart*100:5.2f} %")
    print(f"Accuracy from octopus: {ref_accuracy_octopus*100:5.2f} %")
    print(f"Accuracy from baby: {ref_accuracy_baby*100:5.2f} %")
    
    generate_explained_variance_plot(trainx, SHOW=True)

    print("\n --- \n")


