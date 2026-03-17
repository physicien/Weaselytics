#!/usr/bin/python3

import pandas as pd
import numpy as np
import math
from scipy import stats as st
#from uncertainties_pandas import UncertaintyArray, UncertaintyDtype
#from uncertainties import ufloat
#from uncertainties import unumpy as unp
#import uncertainties.umath as umath
pd.set_option("display.max_rows", None)

#GLOBAL CONSTANTS
DATAPATH = "./solvent_effect_data.csv"
solv_categories = ["Cyclohexane","Benzene","Chlorocyclohexane",\
        "Toluene","XylenesMix","4-Xylene","3-Xylene",\
        "2-Xylene","Chlorobenzene","4-Chlorotoluene","3-Chlorotoluene",\
        "2-Chlorotoluene","3-Dichlorobenzene","2-Dichlorobenzene"]
#solv_categories = ["2-Dichlorobenzene","3-Dichlorobenzene","2-Chlorotoluene",\
#        "3-Chlorotoluene","4-Chlorotoluene","Chlorobenzene","2-Xylene",\
#        "3-Xylene","4-Xylene","XylenesMix","Toluene",\
#        "Chlorocyclohexane","Benzene","Cyclohexane"]
df = pd.read_csv(DATAPATH,comment='#')
df.solvent = pd.Categorical(df.solvent,
                            categories=solv_categories,
                            ordered=True
                            )
mol_categories = ["C60","C70","C90","C90m","C96","C100","t0","CS2","n-Pentane"]
df.mol = pd.Categorical(df.mol,
                        categories=mol_categories,
                        ordered=True
                        )

#FUNCTIONS
#def retention_time(df):
#    x=df["x0"]
#    dx=df["sigma"]
#    return ufloat(x,dx)

def weight(df):
    return df["sigma"].pow(-2)

def weighted_avg_and_std(values, weights):
    """
    Return the weighted standard deviation.

    The weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def avg_retention_time(df):
    df["w"] = weight(df)
    df_restricted = df[["mol","solvent","id","distribution","x0","w"]]
    gb = df_restricted.groupby(["mol","solvent","distribution"],
                               observed=True)[["x0","w"]]
    df = gb.apply(lambda x: weighted_avg_and_std(x["x0"],x["w"])[0])\
            .reset_index(name="x0_wav")
    sigma = gb.apply(lambda x: weighted_avg_and_std(x["x0"],x["w"])[1])\
            .reset_index(name="sigma_wav")
    n_obs = gb.size().to_frame("n").reset_index()
    df.insert(2,"n_obs",n_obs["n"])
    df["sigma_wav"] = sigma["sigma_wav"]
    return df

def chauvenets_criterions(df):
    igb = avg_retention_time(df)
    reject_list = []
    for i in df.index:
        mol = df.iloc[i]["mol"]
        solvent = df.iloc[i]["solvent"]
        distribution = df.iloc[i]["distribution"]
        n_obs = igb[(igb["solvent"]==solvent)&(igb["mol"]==mol)
                    &(igb["distribution"]==distribution)]["n_obs"]
        x_bar = igb[(igb["solvent"]==solvent)&(igb["mol"]==mol)
                    &(igb["distribution"]==distribution)]["x0_wav"]
        sigma_x = igb[(igb["solvent"]==solvent)&(igb["mol"]==mol)
                      &(igb["distribution"]==distribution)]["sigma_wav"]
        x = df.iloc[i]["x0"]
        expt_id = df.iloc[i]["id"]

        #Probability represented by one tail of the normal distribution
        P_z = 1 - (1/(4*n_obs))
        #Maximum allowable deviation
        D_max = st.norm.ppf(P_z)
        #z-score of x
        z_score = abs(x - x_bar)/sigma_x
        reject = z_score > D_max
        reject_list.append(reject.iloc[0])
#        print(f'{i:3}{solvent:>15}{mol:>7}{expt_id:>3}'\
#                f'{str(reject.iloc[0]):>10}{x:>10.3f}{x_bar.iloc[0]:>10.3f}')
    df["reject"] = reject_list
    result = df[~df["reject"]].drop(["w","reject"], axis=1).reset_index()
    return result.drop(["index"], axis=1)

#DATA PROCESSING
#arr_tR = df.apply(retention_time, axis=1)
#df["tR"] = UncertaintyArray(arr_tR)

#print(df)
df_clean = chauvenets_criterions(df)
#print(df_clean)

test = avg_retention_time(df_clean)
print(test[test["distribution"]=="Gaussian"])
print(test[test["distribution"]=="Gaussian"]["n_obs"].sum())
print("")
print(test[test["distribution"]=="Skew-Normal"])
print(test[test["distribution"]=="Skew-Normal"]["n_obs"].sum())

#print(test[(test["distribution"]=="Gaussian") & (test["mol"]=="t0")])
