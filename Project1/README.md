# Dette er prosjekt 1 :)!

# FYS-STK Project 1

The current directory contains everything related to Project 1. This has been a collaboration between [Anna Aasen](https://github.com/Annaaasen), [Carl Martin Fevang](https://github.com/carlmfe) and [Håkon Kvernmoen](https://github.com/hkve). The directory ``src`` contains all the code written throughout this project, while the directory ``tex`` contains everything related to the written report. A pdf report can be found INSERTE WHERE HERE.

The ``src`` directory contains 3 subdirectories (``sknotlearn``, ``analysis`` and ``tests``) and 1 file (``main.py``). 

``sknotlearn`` implements 3 fitting schemes, Ordinary least squares (OLS), Ridge regression and least absolute shrinkage and selection operator (Lasso). Additionally, two resampling schemes are implemented, namely Bootstrapping and k-fold Cross-Validation. This code uses a custom data class where scaling, train test splitting and so on can be preformed.      

``analysis`` is the subdirectory where we actually use this code on two specific datasets, an artificially created hilltop landscape made using the Franke function and digital terrain data from Central America.

``tests`` contains some simple tests of our ``sknotlearn`` code. We do some simple comparisons with ``sklearn``. This can be run using python testframeworks (i.e. *pytest*).

```console
git merge please
```