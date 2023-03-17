import numpy as np

def get_TSSB(name_of_X, path_to_desc="datasets/TSSB/"):

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

def get_divisors(x):
    """Returns a list of divisors of x."""
    divisors = [i for i in range(1, x+1) if x % i == 0]
    return divisors

def get_divisors_for_window_size(x, window_size):
    """Returns the smallest divisor of x greater than or equal to window_size/4."""
    divisors = get_divisors(x)
    for divisor in divisors:
        if divisor >= int(window_size/4):
            return divisor
      
def get_Y(mask):
    """Returns a list of 1s and 0s, where 1 represents a row in mask with a 1, and 0 represents a row with all 0s."""
    Y = [1 if 1 in row else 0 for row in mask]
    return Y

def transform_mask(_X, _mask):
    """Transforms the mask from the shape of for exsample [0,0,1,0,0,0,1,0,0,1,0,1,0,0,0] to [0,0,1,1,1,1,2,2,2,3,3,4,4,4,4] and reparis to [0,0,1,1,1,1,2,2,2,3,3,1,1,1,1] if 1 and 4 were the same""" 
    current_group = 0
    group_indices = []
    for i in range(len(_mask)):
        # print(_mask[i])
        if _mask[i] == 1:
            current_group += 1
        group_indices.append(current_group)
        
    new_mask = np.array(group_indices)
    
    unique_values = np.unique(new_mask)
    total_unique_values = np.unique(unique_values)

    for i in unique_values:
        for j in range(i, unique_values[-1]+1):
            is_equal = _X[[x == i for x in new_mask]][:100] == _X[[x == j for x in new_mask]][:100]            
            if is_equal.all():
                total_unique_values[j] = total_unique_values[i]
                
    for i in unique_values:
        new_mask[i==new_mask] = total_unique_values[i]

    true_unique_values = np.unique(new_mask)
    true_total_unique_values=np.array(range(len(unique_values)))
    
    for i in true_total_unique_values:     
        mask_index = new_mask == true_unique_values[i]
        new_mask[mask_index] = int(i)
        
    return new_mask.reshape(1,-1)[0]

def create_windows_with_stride(_X, mask, window_size):
    stride = get_divisors_for_window_size(int(len(_X) - window_size), window_size)
    num_windows = (len(_X) - window_size) // stride + 1
    reshaped_X = np.array([_X[i * stride:i * stride + window_size] for i in range(num_windows)])

    if mask is None:
        return reshaped_X
    else:
        mask = create_windows(_X = mask, window_size = window_size, mask_is = 0, stride_on = True)
        Y = get_Y(mask)
        return reshaped_X, mask, Y

def create_windows_without_stride(_X, mask, window_size):
    reshaped_X = _X.to_numpy().reshape(-1, 203)
    if mask is None:
        return reshaped_X
    else:
        mask = mask.to_numpy().reshape(-1, 203)
        Y = get_Y(mask)
        return reshaped_X, mask, Y

def create_windows_with_firm_stride(_X, mask, window_size):
    remainder_X, remainder_mask, remainder_True = np.array([]), np.array([]), False
    data_length = len(_X)
    num_windows = data_length // window_size

    # Cut the _X into windows of the specified size
    temp_data = np.array([_X[i*window_size : (i+1)*window_size] for i in range(num_windows)])

    # Reshape the data into an array of shape (-1, window_size)
    reshaped_X = np.array(temp_data).reshape(-1, window_size)
    remainder = _X[num_windows*window_size:]

    if mask is None:
        if len(remainder) > 100:
            remainder_True = True
            remainder_mask = np.append(remainder_mask, remainder)
        return reshaped_X, remainder_mask
    else:
        if len(remainder) > 100:
            remainder_True = True
            remainder_X = np.append(remainder_X, remainder)
        mask, remainder_mask= create_windows(_X = mask, window_size = window_size, mask_is = 0, Firm_stride = True)
        Y = get_Y(mask)
        return reshaped_X, mask, Y, remainder_X, remainder_mask, remainder_True

def create_windows(_X, mask=None, window_size=300, mask_is=1, stride_on=False, Firm_stride=False):
    if stride_on and not Firm_stride:
        return create_windows_with_stride(_X, mask, window_size)
    elif not stride_on and not Firm_stride:
        return create_windows_without_stride(_X, mask, window_size)
    elif Firm_stride:
        return create_windows_with_firm_stride(_X, mask, window_size)

def mask_reshape(mask1):
    for i in range(len(mask1)):
        mask_was_1 = False
        for j in range(len(mask1[i])):
            if mask1[i,j] != 0:
                mask_was_1 = True
            if mask_was_1 == True:
                mask1[i,j] = 1
    return mask1

# def vis_matrix(X_current, vis_type_local):
#     global adj_mat_vis
#     if vis_type_local == "natural":
#         g = NaturalVG(weighted=vis_distance)
#     elif vis_type_local == "horizontal":
#         g = HorizontalVG(weighted=vis_distance)

#     g.build(X_current)

#     adj_mat_vis = np.zeros([len(X_current),len(X_current)], dtype='float')
#     for i in range(len(g.edges)):
#         x, y, q =g.edges[i]
#         adj_mat_vis[x,y] = q
#         if vis_edge_type == "undirected":
#             adj_mat_vis[y,x] = q
        
#         len_x = len(X_current)
#     adj_mat_vis_up=np.append(np.identity(len_x)[1:],[0]*len_x).reshape(len_x,len_x)*adj_mat_vis
#     if vis_edge_type == "undirected":
#         adj_mat_vis_down=np.append([0]*len_x,np.identity(len_x)[:-1]).reshape(len_x,len_x)*adj_mat_vis
#         adj_mat_vis = adj_mat_vis_down + adj_mat_vis_up
#     else:
#         adj_mat_vis = adj_mat_vis_up
#     return adj_mat_vis

#     # functions for creating edge index and edge weight for a given matrix
# def adjToEdgidx(adj_mat):
#     #function for visibility and MTF matrixes
#     edge_index = torch.from_numpy(adj_mat).nonzero().t().contiguous()
#     row, col = edge_index
#     edge_weight = adj_mat[row, col]
#     return edge_index,edge_weight

# def adjToEdgidx_dual_VG(X_current):
#     #function for joined visibility and MTF matrixes
#     pos_adj_mat_vis = vis_matrix(X_current, vis_type)
#     neg_adj_mat_vis = vis_matrix(-X_current, vis_type)

#     edge_index = torch.from_numpy(pos_adj_mat_vis+neg_adj_mat_vis).nonzero().t().contiguous()

#     #join two edge_weight arrays
#     row, col = edge_index
#     edge_weight = np.zeros([len(row),2], dtype='float')
#     edge_weight[:,0] = pos_adj_mat_vis[row, col]
#     edge_weight[:,1] = neg_adj_mat_vis[row, col]
#     return edge_index, edge_weight

# def define_mask(i):
#     if classif == "graph": # for graph classification
#         return torch.tensor(Y[i], dtype=torch.long)
#     elif classif == "node":# for node classification 
#         return torch.unsqueeze(torch.tensor(X_mask[i], dtype=torch.double),1)
