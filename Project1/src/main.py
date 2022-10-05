# Does the actual parsing
import argparse
from pyexpat import model
import numpy as np

import sknotlearn.datasets as datasets
import sknotlearn.linear_model as lm

import analysis.noresampling as nores
import analysis.cross_validation as cv

parser = argparse.ArgumentParser()
plots_parser = parser.add_subparsers(help="Plot", dest="plot")

naive_parser = plots_parser.add_parser("naive")
naive_parser.add_argument("-OLS", "--OLS", type=bool, default=False, help="Use OLS")
naive_parser.add_argument("-R", "--Ridge", type=bool, default=False, help="Use Ridge")
naive_parser.add_argument("-L", "--Lasso", type=bool, default=False, help="Use Lasso")
naive_parser.add_argument("-ds", "--startdeg", type=int, default=1, help="Start degree for plotting MSE and R2")
naive_parser.add_argument("-de", "--enddeg", type=int, default=20, help="End degree for plotting MSE and R2")
naive_parser.add_argument("-ts", "--trainsize", type=float, default=2/3, help="Percentage of data used for training")
naive_parser.add_argument("-rs", "--rndmstate", type=int, default=321, help="Seed used while fitting models")
naive_parser.add_argument("-lR", "--lmbdaRidge", type=float, default=0.1, help="Set lambda value used for Ridge")
naive_parser.add_argument("-lL", "--lmbdaLasso", type=float, default=0.1, help="Set lambda value used for Lasso")
naive_parser.add_argument("-f", "--filename", type=str, default=None, help="Filename in case the plot(s) should be saved. Will chain filename_(type_of_plot)")

cv_parser = plots_parser.add_parser("cv")
cv_parser.add_argument("-k", "--kfolds", nargs="+", type=int, default=7, help="How many folds to run. To compare, type multiple sep by spaces")
cv_parser.add_argument("-OLS", "--OLS", type=bool, default=False, help="Use OLS")
cv_parser.add_argument("-R", "--Ridge", type=bool, default=False, help="Use Ridge")
cv_parser.add_argument("-L", "--Lasso", type=bool, default=False, help="Use Lasso")
cv_parser.add_argument("-ds", "--startdeg", type=int, default=1, help="Start degree for plotting MSE and R2")
cv_parser.add_argument("-de", "--enddeg", type=int, default=20, help="End degree for plotting MSE and R2")
cv_parser.add_argument("-ts", "--trainsize", type=float, default=2/3, help="Percentage of data used for training")
cv_parser.add_argument("-rs", "--rndmstate", type=int, default=321, help="Seed used while fitting models")
cv_parser.add_argument("-lR", "--lmbdaRidge", type=float, default=0.1, help="Set lambda value used for Ridge")
cv_parser.add_argument("-lL", "--lmbdaLasso", type=float, default=0.1, help="Set lambda value used for Lasso")

cv_parser.add_argument("-f", "--filename", type=str, default=None, help="Filename in case the plot(s) should be saved. Will chain filename_(type_of_plot)")

plot_parsers_chained = [
    naive_parser,
    cv_parser
]

for plot_parser in plot_parsers_chained:
    data_parser = plot_parser.add_subparsers(help="Data", dest="data")
    Franke_parser = data_parser.add_parser("Franke")
    Franke_parser.add_argument("-n", "--npoints", type=int, default=600, help="Number of points used to create Franke function dataset")
    Franke_parser.add_argument("-lin", '--linspace', action=argparse.BooleanOptionalAction, default=False, help="Sample from linearly spaced (x,y) values")
    Franke_parser.add_argument("-std", "--stderror", type=float, default=0.1, help="Set the error std for the data set")

    Terrain_parser = data_parser.add_parser("Terrain")
    Terrain_parser.add_argument("-n", "--npoints", type=int, default=900)
    
args = vars(parser.parse_args())
plot = args["plot"]

if args["data"] is None:
    default_data = {
        "data": "Franke", 
        "npoints": 600,
        "linspace": False, 
        "stderror": 0.1   
    }
    
    args.update(default_data)

if args["data"] == "Franke":
    D = datasets.make_FrankeFunction(
        n=args["npoints"],
        linspace=args["linspace"],
        noise_std=args["stderror"],
        random_state=args["rndmstate"]
    )
elif args["data"] == "Terrain":
    D = datasets.load_Terrain(

    )

run_models = [
    args["OLS"],
    args["Ridge"],
    args["Lasso"]
]

# If no model is set, run all
if all(model is False for model in run_models):
    run_models = [True, True, True]

Models = [lm.LinearRegression, lm.Ridge, lm.Lasso]
lmbdas = [None, args["lmbdaRidge"], args["lmbdaLasso"]]
names = ["OLS", "Ridge", "Lasso"]


# Debug
for k, v in args.items():
    print(k,v)

"""
# Preform noresampling functionality
if plot == "naive":
    degrees = np.arange(args["startdeg"], args["enddeg"]+1)
    
    for Model, lmbda, name in zip(Models, lmbdas, names):
        params1, params2 = nores.run_no_resampling(Model, degrees, D, random_state=args["rndmstate"], train_size=args["trainsize"], lmbda=lmbda, noise_std=args["stderror"])
        mse_train, mse_test, r2_train, r2_test = params1

        f1, f2, f3, f4 = None, None, None, None
        if args["filename"]:
            f = f"{args['filename']}_{name}"
            f1, f2 = f+"_mse_noresample", f+"_R2_noresample"
            f3, f4 = f+"OLS_coefs_plots", f+"OLS_coefs_table"

        title = f"{name} no resampling"
        nores.plot_train_test(degrees, mse_train, mse_test, filename=f1)
        nores.plot_train_test(degrees, r2_train, r2_test, filename=f2, ylabel=r"R$^2$-score", title=title)


        if Model == lm.LinearRegression:
            nores.plot_theta_progression(*params2, filename=f3)
            nores.plot_theta_heatmap(*params2, filename=f4)
"""


if plot == "cv":
    ks = 

    mse_across_folds = {}

    for Model, lmbda, name in zip(Models, lmbdas, names):
        for k in ks:
            train_mse, test_mse = cv.run_Kfold_cross_validate(Model, degrees, k=k, random_state=321, lmbda=lmbda)
            mse_across_folds[k] = [train_mse, test_mse]
        
        cv.plot_train_mse_kfold(degrees, mse_across_folds, name, filename=f"{name}_mse_kfold")