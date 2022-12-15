import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix


# Testing johan
import plotly.express as px




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
    
    trainx, testx, trainy, testy = train_test_split(container.x, container.y, test_size=0.2)
    # trainx, valx,  trainy, valy  = train_test_split(trainx,      trainy,      test_size=0.2)
    # print(trainx.head())
    cols = list(trainx.columns)

    # opp_ = cols[ lambda x: x.split["_"][-1] == "ps"]


    # opp_team_stats = trainx.filter(regex="_opp$", axis=1)
    # print(opp_team_stats.head())

    # prev_season_stats = trainx.filter(regex="_ps$", axis=1)
    # print(prev_season_stats.head())

    # other = 

    """
    Plot 1 - Step plot and histogram. 
    """

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


    fig, ax = plt.subplots()
    ax.bar(range(1,n_features_org+1), exp_var, alpha=0.5)
    ax.step(range(1,n_features_org+1),cum_var_ratio)
    ax.axvline(n_features, ls='--')
    
    ax.set_xlabel(r"\# PCA features")
    ax.set_ylabel(r"Variance [\%]")

    plt.show()

    """
    Plot 2 - Feature scatter plots. 
    """

    n_components_to_use = 3

    labels={str(i): f"PC {i+1}" for i in range(n_components_to_use)}

    # pca = PCA(n_components = n_components)
    # principal_components = pca.fit_transform(trainx)

    fig = px.scatter_matrix(
        principal_components[:,:n_components_to_use],
        dimensions=range(n_components_to_use),
        color=trainy,
        labels=labels
    )
    fig.update_traces(diagonal_visible=True)
    fig.show()

    exit()












    # print(cols[60:])
    # for col in cols[50:]:
    #     print(trainx[col].head(3))

    # s = sns.heatmap(trainx[list(cols[50:80:2])].corr())
    # s.set_yticklabels(s.get_yticklabels())
    # s.set_xticklabels(s.get_xticklabels())
    # plt.show()

    scaler = StandardScaler()
    trainx = scaler.fit_transform(trainx)

    cov_mat = np.cov(trainx.T)
    eigvals, eigvecs = np.linalg.eig(cov_mat)
    
    # print(eigvals)
    # print(eigvecs)
    # from IPython import embed; embed()
    best_eigvec = eigvecs[0] 

    a_match = trainx[200]#.loc[trainx["match_id"] == 20190020]
    # cols = trainx.columns

    # for i in range(len(best_eigvec)):
        # print(best_eigvec[i], cols[i], a_match[i])

    num_matches = len(trainx)
    n_features_org = np.shape(trainx)[1]


    print("five")
    five = np.sum(eigvecs[:5], axis=0)
    print(five)

    fig, ax = plt.subplots(figsize=(14,9))
    ax.plot(range(1,len(five)+1), eigvecs[:10].T, lw=0.8)

    plt.show()

    firstn = 14
    df = pd.DataFrame(data=np.real(eigvecs[:firstn]), columns=cols, index=[f"PC{n+1}" for n in range(firstn)])
    plt.figure(figsize=(14,9))
    s = sns.heatmap(df, cmap="gnuplot")
    s.set_yticklabels(s.get_yticklabels(), fontsize=7)
    s.set_xticklabels(s.get_xticklabels(), fontsize=7)
    plt.tight_layout()
    plt.show()




    guessed_results = randomGuesses(num_matches, get_result_distribution())    

    # fig, ax = plt.subplots()
    # ax.scatter(trainx[:,0], trainx[:,1])
    # ax.plot()


    # print(trainy.data)




    ###
    # Plotly code to get scatter plot of principal components.
    ###
    n_components = 3

    labels={str(i): f"PC {i+1}" for i in range(n_components)}

    pca = PCA(n_components = n_components)
    principal_components = pca.fit_transform(trainx)

    fig = px.scatter_matrix(
        principal_components,
        dimensions=range(n_components),
        color=trainy,
        labels=labels
    )
    fig.update_traces(diagonal_visible=True)
    fig.show()


    # fig, ax = plt.subplots()
    # ax.scatter(principal_components[:,0], principal_components[:,1])
    # plt.show()
    
    # print(np.shape(principal_components))
    exp_var = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(exp_var)
    thresh = 0.99
    n_features = list(cum_var_ratio).index(cum_var_ratio[cum_var_ratio >= thresh][0])
    # print(f"Best #features = {n_features}")

    # print("\n", np.shape(principal_components), "\n..")

    principal_data = pd.DataFrame(data=principal_components[:,:n_features])
    
    final_data = pd.concat([principal_data, trainy.reset_index(drop=True)], axis=1)
    print(final_data.head())


    fig, ax = plt.subplots()
    ax.bar(range(1,n_features_org+1), exp_var, alpha=0.5)
    ax.step(range(1,n_features_org+1),cum_var_ratio)
    ax.axvline(n_features, ls='--')
    
    ax.set_xlabel(r"\# PCA features")
    ax.set_ylabel(r"Variance [\%]")

    plt.show()
   

    print("\n --- \n")


