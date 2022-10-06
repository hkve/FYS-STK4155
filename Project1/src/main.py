# Does the actual parsing
import argparse
from fileinput import filename
from pyexpat import model
import numpy as np

import sknotlearn.datasets as datasets
import sknotlearn.linear_model as lm

import analysis.noresampling as nores
import analysis.cross_validation as cv
import analysis.bootstrap as bs
import analysis.ridgelasso as hm


parser = argparse.ArgumentParser()
plots_parser = parser.add_subparsers(help="Plot", dest="plot")

naive_parser = plots_parser.add_parser("naive")
naive_parser.add_argument("-OLS", "--OLS", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use OLS")
naive_parser.add_argument("-R", "--Ridge", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use Ridge")
naive_parser.add_argument("-L", "--Lasso", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use Lasso")
naive_parser.add_argument("-ds", "--startdeg", type=int, default=1, help="Start degree for plotting MSE and R2")
naive_parser.add_argument("-de", "--enddeg", type=int, default=12, help="End degree for plotting MSE and R2")
naive_parser.add_argument("-ts", "--trainsize", type=float, default=2/3, help="Percentage of data used for training")
naive_parser.add_argument("-rs", "--rndmstate", type=int, default=321, help="Seed used while fitting models")
naive_parser.add_argument("-lR", "--lmbdaRidge", type=float, default=0.1, help="Set lambda value used for Ridge")
naive_parser.add_argument("-lL", "--lmbdaLasso", type=float, default=0.1, help="Set lambda value used for Lasso")
naive_parser.add_argument("-f", "--filename", type=str, default=None, help="Filename in case the plot(s) should be saved. Will chain filename_(type_of_plot)")

cv_parser = plots_parser.add_parser("cv")
cv_parser.add_argument("-k", "--kfolds", nargs="+", type=int, default=7, help="How many folds to run. To compare, type multiple sep by spaces")
cv_parser.add_argument("-OLS", "--OLS", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use OLS")
cv_parser.add_argument("-R", "--Ridge", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use Ridge")
cv_parser.add_argument("-L", "--Lasso", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use Lasso")
cv_parser.add_argument("-ds", "--startdeg", type=int, default=1, help="Start degree for plotting MSE")
cv_parser.add_argument("-de", "--enddeg", type=int, default=20, help="End degree for plotting MSE")
cv_parser.add_argument("-ts", "--trainsize", type=float, default=2/3, help="Percentage of data used for training")
cv_parser.add_argument("-rs", "--rndmstate", type=int, default=321, help="Seed used while fitting models")
cv_parser.add_argument("-lR", "--lmbdaRidge", type=float, default=0.1, help="Set lambda value used for Ridge")
cv_parser.add_argument("-lL", "--lmbdaLasso", type=float, default=0.1, help="Set lambda value used for Lasso")
cv_parser.add_argument("-f", "--filename", type=str, default=None, help="Filename in case the plot(s) should be saved. Will chain filename_(type_of_plot)")

heatmap_parser = plots_parser.add_parser("heatmap")
heatmap_parser.add_argument("-k", "--kfolds", nargs="+", type=int, default=7, help="How many folds to run. To compare, type multiple sep by spaces")
heatmap_parser.add_argument("-R", "--Ridge", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use Ridge")
heatmap_parser.add_argument("-L", "--Lasso", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use Lasso")
heatmap_parser.add_argument("-ds", "--startdeg", type=int, default=1, help="Start degree for plotting MSE")
heatmap_parser.add_argument("-de", "--enddeg", type=int, default=15, help="End degree for plotting MSE")
heatmap_parser.add_argument("-ls", "--startlmbda", type=int, default=-9, help="Start log10(lambda) for plotting MSE")
heatmap_parser.add_argument("-le", "--endlmbda", type=int, default=1, help="End log10(lambda) for plotting MSE")
heatmap_parser.add_argument("-nl", "--nlmbda", type=int, default=21, help="Number of lambda values")
heatmap_parser.add_argument("-ts", "--trainsize", type=float, default=2/3, help="Percentage of data used for training")
heatmap_parser.add_argument("-rs", "--rndmstate", type=int, default=321, help="Seed used while fitting models")
heatmap_parser.add_argument("-f", "--filename", type=str, default=None, help="Filename in case the plot(s) should be saved. Will chain filename_(type_of_plot)")
heatmap_parser.add_argument("-d", "--dump", type=str, default=None, help="If the lambda, degree and MSE meshgridsd should be saved to file")
heatmap_parser.add_argument("-l", "--load", type=str, default=None, help="If the lambda, degree and MSE meshgridsd should be loaded from file")


bsrounds_parser = plots_parser.add_parser("bsrounds")
bsrounds_parser.add_argument("-OLS", "--OLS", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use OLS")
bsrounds_parser.add_argument("-R", "--Ridge", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use Ridge")
bsrounds_parser.add_argument("-L", "--Lasso", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use Lasso")
bsrounds_parser.add_argument("-f", "--filename", type=str, default=None, help="Filename in case the plot(s) should be saved. Will chain filename_(type_of_plot)")
bsrounds_parser.add_argument("-df", "--datafilename", type=str, default=None, help="Filename for Ridge and Lasso heatmap, used to find optimal lambda and degrees")
bsrounds_parser.add_argument("-sh", '--showhist', action=argparse.BooleanOptionalAction, default=False, help="Sample from linearly spaced (x,y) values")
bsrounds_parser.add_argument("-rs", "--rndmstate", type=int, default=321, help="Seed used while fitting models")


# bsdegs -> Model, filename (plot_comparison only if all true) and heatmap data, round
bsdegs_parser = plots_parser.add_parser("bsdegs")
bsdegs_parser.add_argument("-OLS", "--OLS", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use OLS")
bsdegs_parser.add_argument("-R", "--Ridge", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use Ridge")
bsdegs_parser.add_argument("-L", "--Lasso", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use Lasso")
bsdegs_parser.add_argument("-f", "--filename", type=str, default=None, help="Filename in case the plot(s) should be saved. Will chain filename_(type_of_plot)")
bsdegs_parser.add_argument("-nr", "--numrounds", type=int, default=400, help="Number of Bootstrap rounds to use for each degree")
bsdegs_parser.add_argument("-df", "--datafilename", type=str, default=None, help="Filename for Ridge and Lasso heatmap, used to find optimal lambda and degrees")
bsdegs_parser.add_argument("-rs", "--rndmstate", type=int, default=321, help="Seed used while fitting models")


bslmbdas_parser = plots_parser.add_parser("bslmbdas")
bslmbdas_parser.add_argument("-R", "--Ridge", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use Ridge")
bslmbdas_parser.add_argument("-L", "--Lasso", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use Lasso")
bslmbdas_parser.add_argument("-f", "--filename", type=str, default=None, help="Filename in case the plot(s) should be saved. Will chain filename_(type_of_plot)")
bslmbdas_parser.add_argument("-nr", "--numrounds", type=int, default=400, help="Number of Bootstrap rounds to use for each degree")
bslmbdas_parser.add_argument("-df", "--datafilename", type=str, default=None, help="Filename for Ridge and Lasso heatmap, used to find optimal lambda and degrees")
bslmbdas_parser.add_argument("-rs", "--rndmstate", type=int, default=321, help="Seed used while fitting models")
bslmbdas_parser.add_argument("-ls", "--startlmbda", type=int, default=-8, help="Start log10(lambda) for plotting MSE")
bslmbdas_parser.add_argument("-le", "--endlmbda", type=int, default=0, help="End log10(lambda) for plotting MSE")
bslmbdas_parser.add_argument("-nl", "--nlmbda", type=int, default=15, help="Number of lambda values")

bs2lmbdas_parser = plots_parser.add_parser("bs2lmbdas")
bs2lmbdas_parser.add_argument("-ds", "--startdeg", type=int, default=1, help="Start degree for plotting MSE")
bs2lmbdas_parser.add_argument("-de", "--enddeg", type=int, default=15, help="End degree for plotting MSE")
bs2lmbdas_parser.add_argument("-R", "--Ridge", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use Ridge")
bs2lmbdas_parser.add_argument("-L", "--Lasso", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use Lasso")
bs2lmbdas_parser.add_argument("-f", "--filename", type=str, default=None, help="Filename in case the plot(s) should be saved. Will chain filename_(type_of_plot)")
bs2lmbdas_parser.add_argument("-nr", "--numrounds", type=int, default=400, help="Number of Bootstrap rounds to use for each degree")
bs2lmbdas_parser.add_argument("-df", "--datafilename", type=str, default=None, help="Filename for Ridge and Lasso heatmap, used to find optimal lambda and degrees")
bs2lmbdas_parser.add_argument("-rs", "--rndmstate", type=int, default=321, help="Seed used while fitting models")


plot_parsers_chained = [
    naive_parser,
    cv_parser,
    heatmap_parser,
    bsrounds_parser,
    bsdegs_parser,
    bslmbdas_parser,
    bs2lmbdas_parser
]

for plot_parser in plot_parsers_chained:
    data_parser = plot_parser.add_subparsers(help="Data", dest="data")
    Franke_parser = data_parser.add_parser("Franke")
    Franke_parser.add_argument("-np", "--npoints", type=int, default=600, help="Number of points used to create Franke function dataset")
    Franke_parser.add_argument("-lin", '--linspace', action=argparse.BooleanOptionalAction, default=False, help="Sample from linearly spaced (x,y) values")
    Franke_parser.add_argument("-std", "--stderror", type=float, default=0.1, help="Set the error std for the data set")

    Terrain_parser = data_parser.add_parser("Terrain")
    Terrain_parser.add_argument("-np", "--npoints", type=int, default=900)
    
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
        n = args["npoints"],
        random_state=args["rndmstate"]
    )

if "OLS" not in args.keys(): args["OLS"] = False
run_models = [
    args["OLS"],
    args["Ridge"],
    args["Lasso"]
]

# If no model is set, run all
if all(model is False for model in run_models):
    run_models = [True, True, True]

Models = [lm.LinearRegression, lm.Ridge, lm.Lasso]
names = ["OLS", "Ridge", "Lasso"]


# Debug
for k, v in args.items():
    print(k,v)


# Make plots where no resampling is used
if plot == "naive":
    lmbdas = [None, args["lmbdaRidge"], args["lmbdaLasso"]]
    degrees = np.arange(args["startdeg"], args["enddeg"]+1)
    
    for Model, lmbda, name, run in zip(Models, lmbdas, names, run_models):
        if not run: continue
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

# Make crossvalidation plots
if plot == "cv":
    lmbdas = [None, args["lmbdaRidge"], args["lmbdaLasso"]]
    degrees = np.arange(args["startdeg"], args["enddeg"]+1)
    ks = np.unique(np.array(args["kfolds"]))
    ks.sort()

    mse_across_folds = {}

    for Model, lmbda, name, run in zip(Models, lmbdas, names, run_models):
        if not run: continue
        for k in ks:
            train_mse, test_mse = cv.run_Kfold_cross_validate(Model, degrees, k=k, random_state=args["rndmstate"], lmbda=lmbda)
            mse_across_folds[k] = [train_mse, test_mse]
        
        f1 = None
        if args["filename"]: f1 = f"{args['filename']}_{name}_mse_kfold"
        cv.plot_train_mse_kfold(degrees, mse_across_folds, name, filename=f1)

# Make heatmap plots
if plot == "heatmap":
    args["OLS"] = False
    lmbdas = np.logspace(args["startlmbda"], args["endlmbda"], args["nlmbda"])
    degrees = np.arange(args["startdeg"], args["enddeg"])

    for Model, lmbda, name, run in zip(Models, lmbdas, names, run_models):
        if not run: continue
        
        if args["load"]:
            degrees_grid, lmbdas_grid, MSEs = hm.load(args["load"])
        else:
            degrees_grid, lmbdas_grid, MSEs = hm.make_mse_grid(
                Model,
                D, 
                degrees,
                lmbdas,
                args["trainsize"],
                args["rndmstate"]
            )

        if args["dump"]:
            hm.dump(f"{args['dump']}_{name}_heatmap", degrees_grid, lmbdas_grid, MSEs)

        f1 = None
        if args["filename"]:
            f1 = f"{args['filename']}_{name}_heatmap"
        hm.plot_heatmap(degrees_grid, lmbdas_grid, MSEs, model=name, filename=f1)


if plot == "bsrounds":
    rounds = np.arange(30, 1000+1, (1001-30)//100)
    BS_lists_rounds = []
    for model, name, run in zip(Models, names, run_models):
        if not run: continue
        if model in [lm.Ridge, lm.Lasso]:
            if not args["datafilename"]:
                print(f"To use {name} you have to give argument -df (--datafilename) as saved to the heatmap command.")
                exit()
            degrees_grid, lmbdas_grid, MSEs = hm.load(f"{args['datafilename']}_{name}_heatmap")
            optimal_lmbda = lmbdas_grid[np.unravel_index(np.argmin(MSEs), MSEs.shape)]
            optimal_degree = degrees_grid[(np.unravel_index(np.argmin(MSEs), MSEs.shape))[0]][0]
        else:
            optimal_degree = 7
            optimal_lmbda = None

        f1 = None
        if args["filename"]:
            f1 = f"{args['filename']}_{name}_bsrounds"
        
        #across rounds:
        BS_list_rounds = bs.bootstrap_across_rounds(D, model, rounds, degree=optimal_degree, lmbda=optimal_lmbda) #Shows that round=400 gives the stabilized state
        BS_lists_rounds.append(BS_list_rounds)
        bs.plot_mse_across_rounds(BS_list_rounds, rounds, model, filename=f1)


if plot == "bsdegs":
    BS_lists_deg = []
    for model, name, run in zip(Models, names, run_models):
        if not run: continue
        degrees = np.arange(1, 15+1)
        optimal_lmbdas = np.array([None]*len(degrees))
        if model in [lm.Ridge, lm.Lasso]:
            if not args["datafilename"]:
                print(f"To use {name} you have to give argument -df (--datafilename) as saved to the heatmap command.")
                exit()
            degrees_grid, lmbdas_grid, MSEs = hm.load(f"{args['datafilename']}_{name}_heatmap")
            optimal_lmbdas = lmbdas_grid[0, np.argmin(MSEs, axis=1)]
            optimal_lmbda = lmbdas_grid[np.unravel_index(np.argmin(MSEs), MSEs.shape)]
            degrees = np.arange(degrees.min(), degrees_grid.max()+1)

        #across degrees:
        BS_list_deg = bs.bootstrap_across_degrees(D, model, args["numrounds"], degrees, lmbdas=optimal_lmbdas)
        BS_lists_deg.append(BS_list_deg)

        f1, f2 = None, None
        if args["filename"]:
            f1 = f"{args['filename']}_{name}_bs_mse"
            f2 = f"{args['filename']}_{name}_bs_biasvar_degs"

        bs.plot_train_test_mse(BS_list_deg, degrees, model)
        bs.plot_bias_var(BS_list_deg, degrees, model)

    if all(run_models):
        bs.plot_comparison(BS_lists_deg)


if plot == "bslmbdas":
    args["OLS"] = False
    lmbdas = np.logspace(args["startlmbda"], args["endlmbda"], args["nlmbda"])
    for model, name, run in zip(Models, names, run_models):
        if not run: continue
        if not args["datafilename"]:
            print(f"To use {name} you have to give argument -df (--datafilename) as saved to the heatmap command.")
            exit()
        degrees_grid, lmbdas_grid, MSEs = hm.load(f"{args['datafilename']}_{name}_heatmap")
        optimal_degree = degrees_grid[(np.unravel_index(np.argmin(MSEs), MSEs.shape))[0]][0]
        BS_list_lam = bs.bootstrap_across_lmbdas(D, lmbdas, model, round=args["numrounds"], degree=optimal_degree)
        
        f1 = None
        if args["filename"]:
            f1 = f"{args['filename']}_{name}_bs_biasvar_lmbdas" 

        bs.plot_bias_var_lmbdas(BS_list_lam, lmbdas, model, filename=f1)


if plot == "bs2lmbdas":
    degrees = np.arange(args["startdeg"], args["enddeg"])
    
    for model, name, run in zip(Models, names, run_models):
        if not run: continue
        if not args["datafilename"]:
            print(f"To use {name} you have to give argument -df (--datafilename) as saved to the heatmap command.")
            exit()
        degrees_grid, lmbdas_grid, MSEs = hm.load(f"{args['datafilename']}_{name}_heatmap")
        optimal_lmbda = lmbdas_grid[np.unravel_index(np.argmin(MSEs), MSEs.shape)] * np.ones_like(degrees)
        bad_boy_lmbda = 1e0 * np.ones_like(degrees)
        BS_R_lmbda1 = bs.bootstrap_across_degrees(D, model, args["numrounds"], degrees, lmbdas=optimal_lmbda)
        BS_R_lmbda2 = bs.bootstrap_across_degrees(D, model, args["numrounds"], degrees, lmbdas=bad_boy_lmbda)
        
        f1 = None
        if args["filename"]:
            f1 = f"{args['filename']}_{name}_bs_biasvar_2lmbdas" 

        bs.plot_bias_var_2lmbda(BS_R_lmbda1, BS_R_lmbda2, optimal_lmbda, bad_boy_lmbda, degrees, model)