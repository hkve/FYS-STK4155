# FYS-STK4155 Project 5 SOMETITLE
### Collaborators: [Carl Martin Fevang](https://github.com/carlmfe), [Anna Aasen](https://github.com/Annaaasen), [Håkon Kvernmoen](https://github.com/hkve), [Nanna Bryne](https://github.com/nannabryne), [Johan Mylius Kroken](https://github.com/johanmkr)

## Abstract (could  also remove  this)
> Abstract of the report placed here-

## Report
The report can be found [here](tex/main.pdf) or in `tex/main.pdf`. 


## Prerequisites
The following packages are required:

* `numpy` $\geq$ 1.17
* `sklearn` Any
* `tensorflow`$\geq$ 2.9
* `pandas` $\geq$ 1.4
* `matplotlib` $\geq$ 3.1 
* `seaborn` $\geq$ 0.12.0
* `imageio` $\geq$ 1.5
* `autograd` $\geq$ 1.5


## Excecution of Code
The code is executed by running scripts in the `analysis` folder. Scripts in the `sknotlearn` and `tensorno` folders are utility files which are not necessary to run on their own. 

Within the `analysis` folder:

In order to tune the LWTA-networks execute:

    python3 tune_LWTA_architecture.py

In order to visualise its activations and/or pathways execute one or both of the following:

    python3 visualise_activation.py
    python3 visualise_network.py

In order to generate the explained variance from the PCA on the EPL data, and show plot, execute:

    python3 pca_EPL1920.py

In order to run the bias-variance analysis of the FIFA2021 dataset execute:

    python3 fifa21.py

Happy execution!

## Structure
some structure, also write something about Carl Martins files (hyperparameter tuning or something).

All figures can be found in `tex/figs`. All code and data files can be found in the `src` folder with the following structure:

* `analysis`
    * `context.py` $\to$ File to redirect paths
    * `fifa21.py` $\to$ Bias-variance tradeoff analysis of FIFA2021 data
    * `fifa21_utils.py` $\to$ Utilities to the above analysis
    * `model_analysis.py` $\to$ Generate model specific plots
    * `network_plot_tools.py` $\to$ Tool for general plotting
    * `pca_EPL1920.py` $\to$ PCA of the EPL data
    * `plot_utils.py` $\to$ Utilities for plotting
    * `tune_LWTA_architecture.py` $\to$ Tune architecture of networks
    * `visualise_activation.py` $\to$ Visualise LWTA activation
    * `visualise_network.py` $\to$ Visualise LWTA pathways

* `sknotlearn`
    * `datasets.py` $\to$ Contains all datasets
    * `fifa21_EAsports_reader.py` $\to$ Read and creates the FIFA2021 dataset
    * `preprocess_EPL_data.py` $\to$ Preprocesses the EPL data
    * `EPL_notes.md` $\to$ Tables describing the features in the EPL dataset
    * `breast_cancer.csv` $\to$ Dataset
    * `EPL1920.csv` $\to$ Dataset
        - Credit: https://www.football-data.co.uk/englandm.php
    * `players_21_subset_csv` $\to$ Dataset
    * `players_21.csv` $\to$ Dataset
    * `salary18.csv` $\to$ Dataset 
        - Credit: https://fbref.com/en/comps/9/2018-2019/2018-2019-Premier-League-Stats
    * `salary19.csv` $\to$ Dataset 
        - Credit: https://fbref.com/en/comps/9/2018-2019/2018-2019-Premier-League-Stats
    * `understat_per_game.csv` $\to$ Dataset
        - Credit: https://www.kaggle.com/datasets/slehkyi/extended-football-stats-for-european-leagues-xg?select=understat_per_game.csv
    
* `tensorno`
    * `bob.py` $\to$ Contains network builders
    * `layers.py` $\to$ Contains LWTA layers
    * `tuner.py` $\to$ Contains LWTA tuners
    * `utils.py` $\to$ Contains LWTA utils





