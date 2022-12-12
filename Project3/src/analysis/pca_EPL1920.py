import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix




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
    print(trainx.head())
    cols = trainx.columns

    scaler = StandardScaler()
    trainx = scaler.fit_transform(trainx)

    cov_mat = np.cov(trainx.T)
    eigvals, eigvecs = np.linalg.eig(cov_mat)
    
    print(eigvals)
    print(eigvecs)
    best_eigvec = eigvecs[0] 

    a_match = trainx[200]#.loc[trainx["match_id"] == 20190020]
    # cols = trainx.columns

    for i in range(len(best_eigvec)):
        print(best_eigvec[i], cols[i], a_match[i])

    num_matches = len(trainx)
    n_features_org = np.shape(trainx)[1]


    print("five")
    five = np.sum(eigvecs[:5], axis=0)
    print(five)

    fig, ax = plt.subplots()

    ax.plot(range(1,len(five)+1), five)

    plt.show()



    guessed_results = randomGuesses(num_matches, get_result_distribution())    

    pca = PCA()
    principal_components = pca.fit_transform(trainx)
    
    # print(np.shape(principal_components))
    cum_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    thresh = 0.99
    n_features = list(cum_var_ratio).index(cum_var_ratio[cum_var_ratio >= thresh][0])
    print(f"Best #features = {n_features}")

    print("\n", np.shape(principal_components), "\n..")

    principal_data = pd.DataFrame(data=principal_components[:,:n_features])
    

    final_data = pd.concat([principal_data, trainy.reset_index(drop=True)], axis=1)
    print(final_data.head())

    fig, ax = plt.subplots()
    ax.step(range(1,n_features_org+1),cum_var_ratio)
    ax.axvline(n_features, ls='--')
    
    ax.set_xlabel(r"\# PCA features")
    ax.set_ylabel(r"Variance [\%]")

    # plt.show()
   

    print("\n --- \n")


