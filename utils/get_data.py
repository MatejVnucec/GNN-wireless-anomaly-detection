import os, sys
import numpy as np
import pandas as pd
import pyedflib
import mne

def get_versions_TSSB(path_to_desc="datasets/TSSB/"):
    """
    Gets the versions of TSSB

    Args:
        path_to_desc: a str that defines the path to the names of versions
    
    Returns:
       versions: a list of possible versions for TSSB
    """
    
    versions = os.listdir(path_to_desc + "TS/")
    versions = list(filter(lambda x: ".ipynb_checkpoints" not in x, versions))
    return versions
    
def get_TSSB(name_of_X, path_to_desc="datasets/TSSB/"):
    """
    Gets the X and mask of a chosen TSSB time series

    Args:
        name_of_X: name of the chosen TS we want and its mask
        path_to_desc: a str that defines the path to the names of versions
    
    Returns:
       _X: a 1D array containing the time steps of the chosen TSSB TS
       mask: a 1D array containing the mask for the chosen TSSB TS
    """
    
    data = []
    with open(path_to_desc+"desc.txt") as f:
        for line in f:
            row = line.strip().split(",")
            data.append(row)

    lists = {}
    for row in data:
        name = row[0]
        values = row[2:]
        lists[name] = values
    _X = np.loadtxt(path_to_desc + "TS/" + name_of_X)
    mask = np.zeros((len(_X)))
    for values in lists[name_of_X[:-4]]:
        mask[int(values)] = 1 
    return _X, mask

def get_Rutgers(DRConfig):
    """
    Gets the X and mask of a chosen TSSB time series

    Args:
        DRConfig: contains the path to the data of rutgers dataset
    
    Returns:
       _X: a 2D array containing the time steps of the rutgers dataset
       mask: a 2D array containing the mask for node classification for the rutgers dataset
       Y: a 1D array containing the mask for graph classification for the rutgers dataset
    """
    
    if DRConfig["len_type"] == "un/cut":
    
        df = pd.read_csv(DRConfig["path_main"])  
        del df['Unnamed: 0']
        df.index, df.columns = [range(df.index.size), range(df.columns.size)]
        length_rss = int((df.columns.stop-2)/2)
        
        X = df.loc[:,df.columns[:length_rss]].to_numpy() # x values for every sample
        #X = np.round_(X,1)
        Y = df[length_rss+1].to_numpy(dtype=np.uint8) # types of anomalies
        X_mask = df.loc[:,df.columns[length_rss+2:]].to_numpy() # binary location of anomalies
        # for i in range(len(Y)):
        #     X_mask[i][X_mask[i] == 1] = Y[i]
        
    # preparation for random graphs
    elif DRConfig["len_type"] == "random":
        dataset_rss = np.load(DRConfig["path_main"], allow_pickle=True)['arr_0']
        dataset_properties = np.load(DRConfig["path_properties"], allow_pickle=True)['arr_0']
        dataset_mask = np.load(DRConfig["path_mask"], allow_pickle=True)['arr_0']

        for i in range(len(dataset_properties)):
            dataset_properties[i,1] = int(dataset_properties[i,1])
        
        X = dataset_rss # x values for every sample
        X_mask = dataset_mask # binary location of anomalies
        Y = dataset_properties[:,2] # types of anomalies
        # Y_len = dataset_properties[:,0] # length of every sample
        
    return X, X_mask, Y

def get_UTime(DRConfig):
    versions = os.listdir("datasets/U-Time/")
    versions.sort()
    version = os.listdir("datasets/U-Time/" + versions[0])
    version.sort()
    
    PSG_file = pyedflib.EdfReader("datasets/U-Time/" + versions[0] +"/"+ version[1]).readAnnotations()
    H_file = pyedflib.EdfReader("datasets/U-Time/" + versions[0] +"/"+ version[0]).readAnnotations()
    