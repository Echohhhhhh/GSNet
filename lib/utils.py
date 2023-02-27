import numpy as np
import pandas as pd
import torch
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from lib.metrics import mask_evaluation_np


class Scaler_NYC:
    def __init__(self, train):
        """ NYC Max-Min
        
        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)
        """
        train_temp = np.transpose(train, (0, 2, 3, 1)).reshape((-1, train.shape[1]))
        self.max = np.max(train_temp, axis=0)
        self.min = np.min(train_temp, axis=0)

    def transform(self, data):
        """norm train，valid，test
        
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)
        
        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T, D, W, H = data.shape
        data = np.transpose(data, (0, 2, 3, 1)).reshape((-1, D))
        data[:, 0] = (data[:, 0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:, 33:40] = (data[:, 33:40] - self.min[33:40]) / (self.max[33:40] - self.min[33:40])
        data[:, 40] = (data[:, 40] - self.min[40]) / (self.max[40] - self.min[40])
        data[:, 46] = (data[:, 46] - self.min[46]) / (self.max[46] - self.min[46])
        data[:, 47] = (data[:, 47] - self.min[47]) / (self.max[47] - self.min[47])
        return np.transpose(data.reshape((T, W, H, -1)), (0, 3, 1, 2))

    def inverse_transform(self, data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)
        
        Returns:
            {np.ndarray} --  shape (T, D, W, H)
        """
        return data * (self.max[0] - self.min[0]) + self.min[0]


class Scaler_Chi:
    def __init__(self, train):
        """Chicago Max-Min
        
        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)
        """
        train_temp = np.transpose(train, (0, 2, 3, 1)).reshape((-1, train.shape[1]))
        self.max = np.max(train_temp, axis=0)
        self.min = np.min(train_temp, axis=0)

    def transform(self, data):
        """norm train，valid，test
        
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)
        
        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T, D, W, H = data.shape
        data = np.transpose(data, (0, 2, 3, 1)).reshape((-1, D))  # (T*W*H,D)
        data[:, 0] = (data[:, 0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:, 33] = (data[:, 33] - self.min[33]) / (self.max[33] - self.min[33])
        data[:, 39] = (data[:, 39] - self.min[39]) / (self.max[39] - self.min[39])
        data[:, 40] = (data[:, 40] - self.min[40]) / (self.max[40] - self.min[40])
        return np.transpose(data.reshape((T, W, H, -1)), (0, 3, 1, 2))

    def inverse_transform(self, data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)
        
        Returns:
            {np.ndarray} --  shape(T, D, W, H)
        """
        return data * (self.max[0] - self.min[0]) + self.min[0]


def mask_loss(predicts, classify_predicts, labels, region_mask, loss_mask, data_type="nyc"):
    """

    Arguments:
        predicts {Tensor} -- predict，(batch_size,pre_len,W,H)
        labels {Tensor} -- label，(batch_size,pre_len,W,H)
        region_mask {np.array} -- mask matrix，(W,H)
        data_type {str} -- nyc/chicago
    
    Returns:
        {Tensor} -- MSELoss,(1,)
    """
    batch_size, pre_len, _, _ = predicts.shape
    region_mask = torch.from_numpy(region_mask).to(predicts.device)
    loss_mask = torch.from_numpy(loss_mask).to(predicts.device)
    region_mask /= region_mask.mean()
    loss = ((labels - predicts) * region_mask) ** 2
    # classification
    classify_labels = labels.view(labels.shape[0], -1)
    # classify_labels = classify_labels.where(classify_labels == 0, torch.ones_like(classify_labels))
    loss2 = bce(classify_predicts, classify_labels)

    # ranking
    # _, classify_labels = torch.sort(labels.view(labels.shape[0], -1))

    # rankingLoss = torch.nn.MarginRankingLoss()
    # print(classify_predicts.shape, classify_labels.shape, target.shape)
    # print(classify_predicts[0, :], classify_labels[0, :], target[0, :])
    # loss2 = listMLE(classify_predicts, classify_labels).requires_grad_(True)
    # loss2 = listNet(classify_predicts, classify_labels).requires_grad_(True)

    # loss2.requires_grad_(True)
    # print(loss2)
    # exit(0)
    if data_type == 'nyc':
        ratio_mask = torch.zeros(labels.shape).to(predicts.device)
        index_1 = labels <= 0
        index_2 = (labels > 0) & (labels <= 0.04)
        index_3 = (labels > 0.04) & (labels <= 0.08)
        index_4 = labels > 0.08
        loss_mask = loss_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, pre_len, 1, 1)
        mask = loss_mask == 0

        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        ratio_mask[mask] = 0

        loss *= ratio_mask
    elif data_type == 'chicago':
        ratio_mask = torch.zeros(labels.shape).to(predicts.device)
        index_1 = labels <= 0
        index_2 = (labels > 0) & (labels <= 1 / 17)
        index_3 = (labels > 1 / 17) & (labels <= 2 / 17)
        index_4 = labels > 2 / 17
        # loss_mask = loss_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, pre_len, 1, 1)
        # mask = loss_mask == 0

        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        # ratio_mask[mask] = 0
        loss *= ratio_mask
    # return torch.mean(loss) + 1e-8 * loss2
    # return torch.mean(loss)
    return torch.mean(loss) + 1e-3 * loss2
    # return loss2

def bce(y_pred, y_true, padded_value_indicator=0):
    """
    Binary Cross-Entropy loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device

    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator

    ls = torch.nn.BCELoss(reduction='none')(y_pred, y_true)
    ls[mask] = 0.0

    document_loss = torch.sum(ls, dim=-1)
    sum_valid = torch.sum(valid_mask, dim=-1).type(torch.float32) > torch.tensor(0.0, dtype=torch.float32, device=device)
    loss_output = torch.sum(document_loss) / torch.sum(sum_valid)

    return loss_output
def listMLE(y_pred, y_true, eps=1e-10, padded_value_indicator=0):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator
    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float('-1e10')
    # print(mask.shape)
    # print(preds_sorted_by_true.shape)
    # exit(0)
    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))

def listNet(y_pred, y_true, eps=1e-10, padded_value_indicator=0):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # y_pred = y_pred.clone()
    # y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-1e10')
    y_true[mask] = float('-1e10')

    preds_smax = torch.softmax(y_pred, dim=1)
    true_smax = torch.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(true_smax * preds_log, dim=1))


@torch.no_grad()
def compute_loss(net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
                 grid_node_map, global_step, device, loss_mask,
                 data_type='nyc'):
    """compute val/test loss
    
    Arguments:
        net {Molde} -- model
        dataloader {DataLoader} -- val/test dataloader
        risk_mask {np.array} -- mask matrix，shape(W,H)
        road_adj  {np.array} -- road adjacent matrix，shape(N,N)
        risk_adj  {np.array} -- risk adjacent matrix，shape(N,N)
        poi_adj  {np.array} -- poi adjacent matrix，shape(N,N)
        global_step {int} -- global_step
        device {Device} -- GPU
    
    Returns:
        np.float32 -- mean loss
    """
    net.eval()
    temp = []
    for feature, target_time, graph_feature, label in dataloader:
        feature, target_time, graph_feature, label = feature.to(device), target_time.to(device), graph_feature.to(
            device), label.to(device)
        final_output, classification_output = net(feature, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                                  grid_node_map)
        l = mask_loss(final_output, classification_output, label, risk_mask, loss_mask, data_type)
        temp.append(l.cpu().item())
    loss_mean = sum(temp) / len(temp)
    return loss_mean


@torch.no_grad()
def predict_and_evaluate(net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
                         grid_node_map, global_step, scaler, device):
    """predict val/test, return metrics
    
    Arguments:
        net {Model} -- model
        dataloader {DataLoader} -- val/test dataloader
        risk_mask {np.array} -- mask matrix，shape(W,H)
        road_adj  {np.array} -- road adjacent matrix，shape(N,N)
        risk_adj  {np.array} -- risk adjacent matrix，shape(N,N)
        poi_adj  {np.array} -- poi adjacent matrix，shape(N,N)
        global_step {int} -- global_step
        scaler {Scaler} -- record max and min
        device {Device} -- GPU
    
    Returns:
        np.float32 -- RMSE，Recall，MAP
        np.array -- label and pre，shape(num_sample,pre_len,W,H)

    """
    net.eval()
    prediction_list = []
    label_list = []
    for feature, target_time, graph_feature, label in dataloader:
        feature, target_time, graph_feature, label = feature.to(device), target_time.to(device), graph_feature.to(
            device), label.to(device)
        final_output, classify_output = net(feature, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                            grid_node_map)
        prediction_list.append(final_output.cpu().numpy())
        label_list.append(label.cpu().numpy())
    prediction = np.concatenate(prediction_list, 0)
    label = np.concatenate(label_list, 0)

    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)

    rmse_, recall_, map_ = mask_evaluation_np(inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    return rmse_, recall_, map_, inverse_trans_pre, inverse_trans_label
