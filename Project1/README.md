# Dette er prosjekt 1 :)!

# FYS-STK Project 1

The current directory contains everything related to Project 1. This has been a collaboration between [Anna Aasen](https://github.com/Annaaasen), [Carl Martin Fevang](https://github.com/carlmfe) and [Håkon Kvernmoen](https://github.com/hkve). The directory ``src`` contains all the code written throughout this project, while the directory ``tex`` contains everything related to the written report. A pdf report can be found INSERT WHERE HERE.

The ``src`` directory contains 3 subdirectories (``sknotlearn``, ``analysis`` and ``tests``) and 1 file (``main.py``). 

``sknotlearn`` implements 3 fitting schemes, Ordinary least squares (OLS), Ridge regression and least absolute shrinkage and selection operator (Lasso). Additionally, two resampling schemes are implemented, namely Bootstrapping and k-fold Cross-Validation. This code uses a custom data class where scaling, train test splitting and so on can be preformed.      

``analysis`` is the subdirectory where we actually use this code on two specific datasets, an artificially created hilltop landscape made using the Franke function and digital terrain data from Central America.

``tests`` contains some simple tests of our ``sknotlearn`` code. We do some simple comparisons with ``sklearn``. This can be run using python test frameworks (i.e. *pytest*).

``main.py`` is the control panel of this code base. It implements a command line interface where the plots shown in the report can be produced. It follows the general syntax

```bash
python3 main.py plotname -optional -plot -args datatype -optional -data -args 
```

To see the list of available plots, together with a short explanation of what these do, execute: 

```bash
python3 main.py --help 
``` 

Take for instance the ``cv`` (Cross-validation) plots. To get a list of tweakable parameters, execute:  
 
```bash
python3 main.py cv --help 
``` 

Let us say we want to plot k = 5, 7 and 10 with OLS, we can specify this by: 

```bash
py main.py cv -k 5 7 10 -OLS
```

After you have added your plot parameters, you can specify the data as either *Franke* or *Terrain* (Franke is default, thus the previous line will produce figure 5 in the report). The data can also be tweaked. If you want to see the possible parameters for terrain data, you can try:

```bash
py main.py cv -k 5 7 10 -OLS Terrain --help
```

Now deciding we want to produce the same plot as with the Franke data, but with 250 points from the digital terrain data, we execute:

```bash
py main.py cv -k 5 7 10 -OLS Terrain -np 250
```
 
The ``main.py`` structure is a mess, I recommend not looking at it if you want to sleep tonight. Therefor running the source files present inside ``analysis`` is of course also possible.  