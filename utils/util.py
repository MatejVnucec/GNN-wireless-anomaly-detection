import numpy as np
import torch
import sklearn
from sklearn.utils import class_weight

def from_array_to_lists(arr):
    """
    This function enables us to transform an array to list or list of arrays. It does this by transforming the first row of the array into lists

    Typically, this function is not called directly, but used by the
    higher-level 'create_notflexible_windows_with_remainder' function.

    Args:
        arr: array of user defined dimensions

    Returns:
        list_of_arrays: a list of arrays/something
    """
    list_of_arrays = [arr[i] for i in range(arr.shape[0])]
    return list_of_arrays

def get_divisors(x):
    """
    Returns a list of divisors of x.

    Typically, this function is not called directly, but used by the
    higher-level 'get_divisors_for_window_size' function.
    
    Args:
        x: a number

    Returns:
        divisors: a list of all posible divisors for x
    """
    divisors = [i for i in range(1, x+1) if x % i == 0]
    return divisors

def get_divisors_for_window_size(x, window_size):
    """
    Returns the smallest divisor of x greater than or equal to window_size/4.

    Args:
        x: an int, usualy defined by the lenght of a time series 
        window_size: an int definig the size of desired length the x should be cut to 

    Returns:
        devisor: the devisor that suits the functions criteria
    """
    divisors = get_divisors(x)
    for divisor in divisors:
        if divisor >= int(window_size/4):
            return divisor
    
    raise ValueError("Error: no sutable devisor for given length of time series = {x} and window size = {window_size/4}")
      
def get_Y(mask):
    """
    Transforms a mask into an Y where mask values are classified for all time stamps, but Y has only one value per time series

    Args:
        mask: a 2D array of 1s and 0s where a row indicates a single time sereis with time steps

    Returns:
        Y: a list of ints of 0s and 1s, 0 indicating there is no 1 in a row and 1 if there is at least one 1 in that row
    """
    Y = [1 if 1 in row else 0 for row in mask]
    return Y

def transform_mask(_X, _mask):
    """
    Transforms the mask from the shape of for exsample [0,0,1,0,0,0,1,0,0,1,0,1,0,0,0] to [0,0,1,1,1,1,2,2,2,3,3,4,4,4,4] and reparis to [0,0,1,1,1,1,2,2,2,3,3,1,1,1,1] if 1 and 4 were the same

    Args:
        _X: a 1D array containing time series, is used to check if _X is repeating anywere inside it
        _mask : a 1D array that will be converted from 1 and 0 to multi lable mask
    Returns:
        new_mask: a 1D array where the mask is converted from 1 and 0 to multiple lable mask, it is mostly used for time seres segmentation
    """
    
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

def create_flexible_windows(_X, mask, window_size):
    """
    creates a reshaped_X with no remainder, for exsample [0,1,2,3,4,5] we have a window size of 4 it will return reshaped_X =[[0,1,2,3][2,3,4,5]] and will do the same to mask
    
    Typically, this function is not called directly, but used by the higher-level 'create_windows' function.
    
    Args:
        _X: a 1D array
        mask : a 1D array
        window_size: an int refering to the size of newly created X and mask
    Returns:
        reshaped_X: a 2D array with the shape of (window_size, devisors)
        reshaped_mask: a 2D array with the same shape as reshaped_X
        Y: a list of ints created from reshaped_mask using get_Y function
    """
    
    stride = get_divisors_for_window_size(int(len(_X) - window_size), window_size)
    num_windows = (len(_X) - window_size) // stride + 1
    reshaped_X = np.array([_X[i * stride:i * stride + window_size] for i in range(num_windows)])

    if mask is None:
        return reshaped_X
    else:
        reshaped_mask = create_windows(_X = mask, window_size = window_size, stride_type = "flexible")
        Y = get_Y(reshaped_mask)
        return reshaped_X, reshaped_mask, Y

def create_notflexible_windows(_X, mask, window_size):
    """
    creates a reshaped_X. Is only possible if window_size is correctly selected
    
    Typically, this function is not called directly, but used by the higher-level 'create_windows' function.
    
    Args:
        _X: a 1D array 
        mask : a 1D array
        window_size: an int refering to the size of newly created X and mask
    Returns:
        reshaped_X: a 2D array with the shape of (window_size, devisors)
        reshaped_mask: a 2D array with the same shape as reshaped_X
        Y: a list of ints created from reshaped_mask using get_Y function
    """
    
    if _X.size % window_size != 0:
        raise ValueError("Error: X cannot be reshaped by a factor of window_size = {window_size}")

    reshaped_X = _X.to_numpy().reshape(-1, window_size)
    if mask is None:
        return reshaped_X
    else:
        mask = mask.to_numpy().reshape(-1, window_size)
        Y = get_Y(mask)
        return reshaped_X, mask, Y

def create_notflexible_windows_with_remainder(_X, mask, window_size, min_remainder):
    """
    creates a reshaped_X with remainder and no stride, for exsample [0,1,2,3,4,5] we have a window size of 4 it will return reshaped_X = [[0,1,2,3]], remainder_X = [4,5]
    
    Typically, this function is not called directly, but used by the higher-level 'create_windows' function.
    
    Args:
        _X: a 1D array 
        mask : a 1D array
        window_size: an int refering to the size of newly created X and mask
        min_remainder: an int of minimum length of tolerance that we still include the remainder inside of newly reshaped _X
    Returns:
        reshaped_X: a list of arrays if remainder is included, else an 2D array with the shape of (window_size, (len(_X)/window size))
        reshaped_mask: a list of arrays if remainder is included, else an 2D array with the same shape as reshaped_X
        Y: a list of ints created from reshaped_mask using get_Y function
    """
    remainder_X, remainder_mask, remainder_True = np.array([]), np.array([]), False
    data_length = len(_X)
    if window_size > data_length:
        raise ValueError("Error: length of X (=" +seq(data_length) + ") can't be smaller than window_size("+seq(window_size)+"). Please lower the window_size variable or change X")
    num_windows = data_length // window_size

    # Cut the _X into windows of the specified size
    temp_data = np.array([_X[i*window_size : (i+1)*window_size] for i in range(num_windows)])

    # Reshape the data into an array of shape (-1, window_size)
    reshaped_X = np.array(temp_data).reshape(-1, window_size)
    remainder = _X[num_windows*window_size:]

    if mask is None:
        if len(remainder) > min_remainder:
            remainder_True = True
            remainder_mask = np.append(remainder_mask, remainder)
        return reshaped_X, remainder_mask
    else:
        if len(remainder) > min_remainder:
            remainder_True = True
            remainder_X = np.append(remainder_X, remainder)
        mask, remainder_mask= create_windows(_X = mask, window_size = window_size, stride_type = "notflexible_with_remainder")
        if remainder_True == True:
            reshaped_X=from_array_to_lists(reshaped_X)
            mask=from_array_to_lists(mask)
            reshaped_X.append(remainder_X)
            mask.append(remainder_mask)
        Y = get_Y(mask)
        return reshaped_X, mask, Y#, remainder_X, remainder_mask, remainder_True

def create_windows(_X, mask=None, window_size=300, stride_type="notflexible_with_remainder", min_remainder = 100):
    """
    creates a reshaped_X with remainder and no stride, for exsample [0,1,2,3,4,5] we have a window size of 4 it will return reshaped_X = [[0,1,2,3]], remainder_X = [4,5]
    
    Args:
        _X: a 1D array 
        mask : is a 1D array, it is given as None as this i crutial for function recursion
        window_size: an int refering to the size of newly created X and mask
        stride_type: is an string that tells us which of three functions are going to be called later
        min_remainder: an int of minimum length of tolerance that we still include the remainder inside of newly reshaped _X
    Returns:
        reshaped_X: a list of arrays if remainder is included, else an 2D array with the shape of (window_size, (len(_X)/window size))
        reshaped_mask: a list of arrays if remainder is included, else an 2D array with the same shape as reshaped_X
        Y: a list of ints created from reshaped_mask using get_Y function
    """
    
    if stride_type == "flexible":
        return create_flexible_windows(_X, mask, window_size)
    elif stride_type == "notflexible":
        return create_notflexible_windows(_X, mask, window_size)
    elif stride_type == "notflexible_with_remainder":
        return create_notflexible_windows_with_remainder(_X, mask, window_size, min_remainder)

def mask_reshape(mask):
    """ 
    fils in the mask with ones when 1 appeares for exsample [0,0,1,0,0] is converted to [0,0,1,1,1]

    Args:
        mask : a 1D array that will be converted 
    Returns:
        mask: a 1D array where there are 1 once a 1 apears 
    """
    
    for i in range(len(mask)):
        mask_was_1 = False
        for j in range(len(mask[i])):
            if mask[i,j] != 0:
                mask_was_1 = True
            if mask_was_1 == True:
                mask[i,j] = 1
    return mask

def class_weigth(MConfig,_output):
    """ 
    creates a class_weight that helps us with a more balanced training if one anomaly for exsamples apeares more times in a time series it will get less attention than the one that apeares less times

    Args:
        _output : a list of Data of graphs in witch there are Y which are masks we want
    Returns:
        _class_weights: a class_weight that consists of len of classes and a float value vhere higher the value the more a value is found in Y array
    """
    
    all_x = np.array([])
    for obj in _output:
        all_x = np.append(all_x,obj.y.numpy())
    all_x = np.array(all_x).reshape(-1)
    # print(np.unique(all_x))
    class_weights = torch.tensor(class_weight.compute_class_weight(class_weight='balanced',
                                                                    classes=np.unique(all_x),
                                                                    y=all_x))
    
    if MConfig['loss'] == "BCE":
        class_weights =torch.tensor([class_weights[1]/class_weights[0]])
    return class_weights

def test_val_train_split(TConfig,len_graph):
    """ 
    creates a train val test split for the lenght of graph/time series/X

    Args:
        TConfig: a lib that contains the percentage of train val test values we want per given lenght of graph
        len_graph: the length of graph/time series/X
    Returns:
        train_size: an int of desired len for train 
        val_size: an int of desired len for validation
        test_size: an int of desired len for test
    """
    
    
    train_size = int(TConfig["train"] * len_graph)
    Temp_size = len_graph - train_size
    val_size = int(TConfig["val"]*Temp_size)
    test_size = Temp_size - val_size
    
    return train_size, val_size, test_size