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
        train_temp = np.transpose(train,(0,2,3,1)).reshape((-1,train.shape[1]))
        self.max = np.max(train_temp,axis=0)
        self.min = np.min(train_temp,axis=0)
    def transform(self, data):
        """norm train，valid，test
        
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)
        
        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T,D,W,H = data.shape
        data = np.transpose(data,(0,2,3,1)).reshape((-1,D))
        data[:,0] = (data[:,0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:,33:40] = (data[:,33:40] - self.min[33:40]) / (self.max[33:40] - self.min[33:40])
        data[:,40] = (data[:,40] - self.min[40]) / (self.max[40] - self.min[40])
        data[:,46] = (data[:,46] - self.min[46]) / (self.max[46] - self.min[46])
        data[:,47] = (data[:,47] - self.min[47]) / (self.max[47] - self.min[47])
        return np.transpose(data.reshape((T,W,H,-1)),(0,3,1,2))
    
    def inverse_transform(self,data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)
        
        Returns:
            {np.ndarray} --  shape (T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]


class Scaler_Chi:
    def __init__(self, train):
        """Chicago Max-Min
        
        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)
        """
        train_temp = np.transpose(train,(0,2,3,1)).reshape((-1,train.shape[1]))
        self.max = np.max(train_temp,axis=0)
        self.min = np.min(train_temp,axis=0)
    def transform(self, data):
        """norm train，valid，test
        
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)
        
        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T,D,W,H = data.shape
        data = np.transpose(data,(0,2,3,1)).reshape((-1,D))#(T*W*H,D)
        data[:,0] = (data[:,0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:,33] = (data[:,33] - self.min[33]) / (self.max[33] - self.min[33])
        data[:,39] = (data[:,39] - self.min[39]) / (self.max[39] - self.min[39])
        data[:,40] = (data[:,40] - self.min[40]) / (self.max[40] - self.min[40])
        return np.transpose(data.reshape((T,W,H,-1)),(0,3,1,2))
    
    def inverse_transform(self,data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)
        
        Returns:
            {np.ndarray} --  shape(T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]


def mask_loss(predicts,labels,region_mask,data_type="nyc"):
    """
    
    Arguments:
        predicts {Tensor} -- predict，(batch_size,pre_len,W,H)
        labels {Tensor} -- label，(batch_size,pre_len,W,H)
        region_mask {np.array} -- mask matrix，(W,H)
        data_type {str} -- nyc/chicago
    
    Returns:
        {Tensor} -- MSELoss,(1,)
    """
    batch_size,pre_len,_,_ = predicts.shape
    region_mask = torch.from_numpy(region_mask).to(predicts.device)
    region_mask /= region_mask.mean()
    loss = ((labels-predicts)*region_mask)**2
    if data_type=='nyc':
        ratio_mask = torch.zeros(labels.shape).to(predicts.device)
        index_1 = labels <=0
        index_2 = (labels > 0) & (labels <= 0.04)
        index_3 = (labels > 0.04) & (labels <= 0.08)
        index_4 = labels > 0.08
        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        loss *= ratio_mask
    elif data_type=='chicago':
        ratio_mask = torch.zeros(labels.shape).to(predicts.device)
        index_1 = labels <=0
        index_2 = (labels > 0) & (labels <= 1/17)
        index_3 = (labels > 1/17) & (labels <= 2/17)
        index_4 = labels > 2/17
        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        loss *= ratio_mask
    return torch.mean(loss)


@torch.no_grad()
def compute_loss(net,dataloader,risk_mask,road_adj,risk_adj,poi_adj,
                grid_node_map,global_step,device,
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
    for feature,target_time,graph_feature,label in dataloader:
        feature,target_time,graph_feature,label = feature.to(device),target_time.to(device),graph_feature.to(device),label.to(device)
        l = mask_loss(net(feature,target_time,graph_feature,road_adj,risk_adj,poi_adj,grid_node_map),label,risk_mask,data_type)#l的shape：(1,)
        temp.append(l.cpu().item())
    loss_mean = sum(temp) / len(temp)
    return loss_mean

@torch.no_grad()
def predict_and_evaluate(net,dataloader,risk_mask,road_adj,risk_adj,poi_adj,
                        grid_node_map,global_step,scaler,device):
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
    for feature,target_time,graph_feature,label in dataloader:
        feature,target_time,graph_feature,label = feature.to(device),target_time.to(device),graph_feature.to(device),label.to(device)
        prediction_list.append(net(feature,target_time,graph_feature,road_adj,risk_adj,poi_adj,grid_node_map).cpu().numpy())
        label_list.append(label.cpu().numpy())
    prediction = np.concatenate(prediction_list, 0)
    label = np.concatenate(label_list, 0)

    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)

    rmse_,recall_,map_ = mask_evaluation_np(inverse_trans_label,inverse_trans_pre,risk_mask,0)
    return rmse_,recall_,map_,inverse_trans_pre,inverse_trans_label