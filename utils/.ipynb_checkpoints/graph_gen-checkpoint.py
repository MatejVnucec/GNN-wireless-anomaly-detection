import numpy as np
from ts2vg import NaturalVG
from ts2vg import HorizontalVG
from pyts.image import MarkovTransitionField
import torch
from torch_geometric.data import Data

def get_matrix(X_current, GConfig):
    """
    This function gets the adjacency matrix through either visibility, MTF or dual VG graph

    Args:
        X_current: a 1D array usualy containing time sereis values
        GConfig: a lib containing all needed strings for determening parameters as: weighted, edge_type, edge_dir, num_bins, type

    Returns:
        adj_mat: an array of shape (len(X_current), len(X_current))
    """
    adj_mat = []
    if GConfig["type"] in ("VG","Dual_VG")  :
        VGConfig=GConfig["VG"]
        
        if VGConfig["edge_type"] == "natural":
            g = NaturalVG(weighted=VGConfig["distance"])
        elif VGConfig["edge_type"] == "horizontal":
            g = HorizontalVG(weighted=VGConfig["distance"])

        g.build(X_current)

        adj_mat_vis = np.zeros([len(X_current),len(X_current)], dtype='float')
        for i in range(len(g.edges)):
            x, y, q =g.edges[i]
            adj_mat_vis[x,y] = q
            if VGConfig["edge_dir"] == "undirected":
                adj_mat_vis[y,x] = q

        # len_x = len(X_current)
        # adj_mat_vis_up=np.append(np.identity(len_x)[1:],[0]*len_x).reshape(len_x,len_x)*adj_mat_vis
        # if VGConfig["edge_dir"] == "undirected":
        #     adj_mat_vis_down=np.append([0]*len_x,np.identity(len_x)[:-1]).reshape(len_x,len_x)*adj_mat_vis
        #     adj_mat_vis = adj_mat_vis_down + adj_mat_vis_up
        # else:
        #     adj_mat_vis = adj_mat_vis_up
        adj_mat.append(adj_mat_vis)
        
    elif GConfig["type"] == "MTF":
        n_bins = GConfig["MTF"]["num_bins"]
        if n_bins == "auto":
            n_bins = len(X_current)
        MTF = MarkovTransitionField(n_bins=n_bins)
        X_gaf_MTF_temp = MTF.fit_transform(X_current.reshape(1, -1))
        adj_mat.append(X_gaf_MTF_temp[0])
    
    return adj_mat

def adjToEdgidx(adj_mat):
    """
    This function creates a edge indexes and weights for given matrix
    
    Args:
        adj_mat: a 2D array

    Returns:
        edge_index: gives a 2D torch array that tells us what values are considered conected 
        edge_weight: gives a 2D array of weights that tell us the absolute distance between every node or value in the time series
    """
    edge_index = torch.from_numpy(adj_mat[0]).nonzero().t().contiguous()
    row, col = edge_index
    edge_weight = adj_mat[0][row, col]
    return edge_index,edge_weight

def adjToEdgidx_dual_VG(X_current, GConfig):
    """
    This function creates a dual visibility graph by first creating a directed VG from one side and then fliping and runing the get_martix function again.
    By doing this we then join this two graphs and get a dual VG

    Args:
        X_current: a 1D array usualy containing time series values
        GConfig: a lib containing all needed strings for determening parameters in the get_matrix function

    Returns:
        edge_index: gives a 2D torch array that tells us what values are considered conected 
        edge_weight: gives a 2D array of weights that tell us the absolute distance between every node or value in time series
    """
    
    
    pos_adj_mat_vis = get_matrix(X_current, GConfig)
    neg_adj_mat_vis = get_matrix(-X_current, GConfig)

    edge_index = torch.from_numpy(pos_adj_mat_vis+neg_adj_mat_vis).nonzero().t().contiguous()

    #join two edge_weight arrays
    row, col = edge_index
    edge_weight = np.zeros([len(row),2], dtype='float')
    edge_weight[:,0] = pos_adj_mat_vis[row, col]
    edge_weight[:,1] = neg_adj_mat_vis[row, col]
    return edge_index, edge_weight

def define_mask(X_mask, Y, GConfig):
    """
    decides if it will use graph of node classification upon checkin the GConfig["classif"]

    Args:
        X_mask: a 1D array
        Y: an int value
        GConfig: a lib containing a string determening the classification type

    Returns:
        either a torch value of Y
        or a torch array of X_mask
    """
    
    if GConfig["classif"] == "graph":
        return torch.tensor(Y, dtype=torch.long)
    elif GConfig["classif"] == "node":# for node classification 
        return torch.unsqueeze(torch.tensor(X_mask, dtype=torch.double),1)

def output_append(GConfig, _X,X_mask=[],Y=[], output=[]):
    """
    By using get_matrix and define_mask we can define a graph and put it into output

    Args:
        GConfig: a lib containing all needed strings for determening parameters in the get_matrix function
        _X: a 2D array usualy containing time series values
        X_mask: a 2D array 
        Y: a 1D array 
        output: an empty list, or it can be defined and this function will append to the imput output list
        
    Returns:
        output: a list of defined graphs compacted into Data and values of (x, edge index,edge atribute, y)
        
    """
    for i in range(len(_X)):
        if GConfig["type"] in ("MTF","VG"): 
            edge_index, edge_weight = adjToEdgidx(get_matrix(_X[i],GConfig))
        elif GConfig["type"] == "dual_VG":
            edge_index, edge_weight = adjToEdgidx_dual_VG(_X[i],GConfig)
        x =  torch.unsqueeze(torch.tensor(_X[i], dtype=torch.double), 1).clone().detach()
        edge_index = edge_index.clone().detach().to(torch.int64)
        edge_attr = torch.unsqueeze(torch.tensor(edge_weight, dtype=torch.double),1).clone().detach()
        y = define_mask(X_mask[i], Y[i], GConfig)

        output.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))   
    return output