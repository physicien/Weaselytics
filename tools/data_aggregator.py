#!/usr/bin/python3
import pandas as pd
import argparse
import os
import re

#create parser
parser = argparse.ArgumentParser(prog='data_aggregator',\
        description='Parse data from all .csv files to easily aggreagate them')

#file is required
parser.add_argument("filename",
    nargs='+',
    help="the .csv input file")

#parse arguments
args = parser.parse_args()

chromato_list=list()

for index,path in enumerate(args.filename):
    filename_path = path
    filename = os.path.basename(path)
    chromato = pd.read_csv(path)
    chromato["id"] = re.match(r"^.+__(\d+)_\w+.csv",filename).group(1)
    chromato_list.append(chromato)

df = pd.concat(chromato_list,ignore_index=True)
first_col = df.pop("id")
df.insert(2, "id", first_col)
df.to_csv("solvent_effect_data.csv", index=False)
