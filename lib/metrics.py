import numpy as np
import math


def transfer_dtype(y_true,y_pred):
    return y_true.astype('float32'),y_pred.astype('float32')

def mask_mse_np(y_true,y_pred,region_mask,null_val=None):
    """
    Arguments:
        y_true {np.ndarray} -- shape (samples,pre_len,W,H)
        y_pred {np.ndarray} -- shape (samples,pre_len,W,H)
        region_mask {np.ndarray} -- mask matrix,shape (W,H)
    
    Returns:
        np.float32 -- MSE
    """
    y_true,y_pred = transfer_dtype(y_true,y_pred)
    if null_val is not None:
        label_mask = np.where(y_true > 0,1,0).astype('float32')
        mask = region_mask * label_mask
    else:
        mask = region_mask
    mask /= mask.mean()
    return np.mean(((y_true-y_pred)*mask)**2)

def mask_rmse_np(y_true,y_pred,region_mask,null_val=None):
    """
    Arguments:
        y_true {np.ndarray} -- shape (samples,pre_len,W,H)
        y_pred {np.ndarray} -- shape (samples,pre_len,W,H)
        region_mask {np.ndarray} -- mask matrix, shape (W,H)
    
    Returns:
        np.float32 -- RMSE
    """
    y_true,y_pred = transfer_dtype(y_true,y_pred)
    return math.sqrt(mask_mse_np(y_true,y_pred,region_mask,null_val))

def nonzero_num(y_true):
    """get the grid number of have traffic accident in all time interval
    
    Arguments:
        y_true {np.array} -- shape:(samples,pre_len,W,H)
    Returns:
        {list} -- (samples,)
    """
    nonzero_list = []
    threshold = 0
    for i in range(len(y_true)):
        non_zero_nums = (y_true[i] > threshold).sum()
        nonzero_list.append(non_zero_nums)
    return nonzero_list

def get_top(data,accident_nums):
    """get top-K risk grid
    Arguments:
        data {np.array} -- shape (samples,pre_len,W,H)
        accident_nums {list} -- (samples,)，grid number of have traffic accident in all time intervals
    Returns:
        {list} -- (samples,k)
    """
    data = data.reshape((data.shape[0],-1))
    topk_list = []
    for i in range(len(data)):
        risk = {}
        for j in range(len(data[i])):
            risk[j] = data[i][j]
        k = int(accident_nums[i])
        topk_list.append(list(dict(sorted(risk.items(),key=lambda x:x[1],reverse=True)[:k]).keys()))
    return topk_list


def Recall(y_true,y_pred,region_mask):
    """
    Arguments:
        y_true {np.ndarray} -- shape (samples,pre_len,W,H)
        y_pred {np.ndarray} -- shape (samples,pre_len,W,H)
        region_mask {np.ndarray} -- mask matrix,shape (W,H)
    Returns:
        float -- recall
    """
    region_mask = np.where(region_mask >= 1,0,-1000)
    tmp_y_true = y_true + region_mask
    tmp_y_pred = y_pred + region_mask

    accident_grids_nums = nonzero_num(tmp_y_true)
    
    true_top_k = get_top(tmp_y_true,accident_grids_nums)
    pred_top_k = get_top(tmp_y_pred,accident_grids_nums)
    
    hit_sum = 0
    for i in range(len(true_top_k)):
        intersection = [v for v in true_top_k[i] if v in pred_top_k[i]]
        hit_sum += len(intersection)
    return hit_sum / sum(accident_grids_nums) * 100


def MAP(y_true,y_pred,region_mask):
    """
        y_true {np.ndarray} -- shape (samples,pre_len,W,H)
        y_pred {np.ndarray} -- shape (samples,pre_len,W,H)
        region_mask {np.ndarray} -- mask matrix,shape (W,H)
    """
    region_mask = np.where(region_mask >= 1,0,-1000)
    tmp_y_true = y_true + region_mask
    tmp_y_pred = y_pred + region_mask
    
    accident_grids_nums = nonzero_num(tmp_y_true)
    
    true_top_k = get_top(tmp_y_true,accident_grids_nums)
    pred_top_k = get_top(tmp_y_pred,accident_grids_nums)

    all_k_AP = []
    for sample in range(len(true_top_k)):
        all_k_AP.append(AP(list(true_top_k[sample]),list(pred_top_k[sample])))
    return sum(all_k_AP)/len(all_k_AP)

def AP(label_list,pre_list):
    hits = 0
    sum_precs = 0
    for n in range(len(pre_list)):
        if pre_list[n] in label_list:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / len(label_list)
    else:
        return 0

def mask_evaluation_np(y_true,y_pred,region_mask,null_val=None):
    """RMSE，Recall，MAP
    
    Arguments:
        y_true {np.ndarray} -- shape (samples,pre_len,W,H)
        y_pred {np.ndarray} -- shape (samples,pre_len,W,H)
        region_mask {np.ndarray} -- mask matrix,shape (W,H)
    Returns:
        np.float32 -- MAE,MSE,RMSE
    """
    rmse_ = mask_rmse_np(y_true,y_pred,region_mask,null_val)
    recall_ = Recall(y_true,y_pred,region_mask)
    map_ = MAP(y_true,y_pred,region_mask)

    return rmse_,recall_,map_