#!/usr/bin/python3
import pandas as pd
import argparse

pd.set_option('display.max_rows', None)

#create parser
parser = argparse.ArgumentParser(prog='data_viz',\
        description='Parse data from .csv files to easily visualize them')

#file is required
parser.add_argument("filename",
    nargs='+',
    help="the .csv input file")

#parse arguments
args = parser.parse_args()

file_list=list()

for index,path in enumerate(args.filename):
    filename_path = path
    file = pd.read_csv(path)
    file_list.append(file)

df = pd.concat(file_list,ignore_index=True)

#print(df[df['mol']=='t0'][['solvent','x0']])
#print(df[df['mol']=='t0'][['x0']].mean())
print(df[df['mol']=='t0'][['solvent','x0']])
