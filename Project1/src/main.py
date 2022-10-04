# Does the actual parsing
import argparse
from ast import parse

import sknotlearn.datasets as datasets
import analysis.noresampling as noresampling


parser = argparse.ArgumentParser()
plots_parser = parser.add_subparsers(help="Plot", dest="plot")

naive_parser = plots_parser.add_parser("naive")
naive_parser.add_argument("-d0", "--degrees0", type=int, default=10)
naive_parser.add_argument("-d1", "--degrees1", type=int, default=10)

plot_parser_chained = [
    naive_parser,
]

for plot_parser in plot_parser_chained:
    data_parser = plot_parser.add_subparsers(help="Data", dest="data")
    Franke_parser = data_parser.add_parser("Franke")
    Franke_parser.add_argument("-n", "--npoints", type=int, default=600)
    
    Terrain_parser = data_parser.add_parser("Franke")
    Terrain_parser.add_argument("-n", "--npoints", type=int, default=900)
    


# parser = argparse.ArgumentParser()
# group = parser.add_mutually_exclusive_group()
# group.add_argument("-F", "--Franke", action="store_const", dest="data", const="Franke")
# group.add_argument("-T", "--Terrain", action="store_const", dest="data", const="Terrain")

# print(group)
# subparser = parser.add_subparsers(help="commands", dest="command")
# noresample_parser = subparser.add_parser("naive", help="Preforms analysis based on a simple train/test split")


# parser.set_defaults(data="Franke")

args = vars(parser.parse_args())
# command = args["command"]

print(args)
