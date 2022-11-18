# FYS-STK Project 1

The current directory contains everything related to Project 2. This has been a collaboration between [Anna Aasen](https://github.com/Annaaasen), [Carl Martin Fevang](https://github.com/carlmfe) and [Håkon Kvernmoen](https://github.com/hkve). The directory ``src`` contains all the code written throughout this project, while the directory ``tex`` contains everything related to the written report. A pdf report can also be found in the `tex` directory.

The ``src`` directory contains two subdirectories ``sknotlearn`` and ``analysis``. ``sknotlearn`` contains the methods implemented for gradient decent, neural network and logistic regression, while ``analysis`` contains the usage of these methods. The following python packages are required:

* ``numpy``, >= 1.17
* ``sklearn``, Any
* ``matplotlib``, >= 3.1
* ``seaborn``, >= 0.12.0
* ``imageio``, >=2.22.1
* ``autograd``, >= 1.5

``sknotlearn``, in addition to the functionality implemented during [Project 1](https://github.com/hkve/FYS-STK4155/tree/main/Project1), contains ``optimize.py`` for gradient decent algorithms, ``neuralnet.py``for our neural network implementation and ``logreg.py`` for our logistic regression implementation.

Plots presented in the report can be recreated using code found in the ``analysis`` directory. Using ``gradient_descent_plots.py``, the plots related to gradient decent can be made. Through ``neuralnet_reg.py``, all plots related to neural network regression can be found. Lastly, the plots presented for classification can be run using ``classification.py``.  