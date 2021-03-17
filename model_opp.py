# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:26:09 2020

@author: 82045
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

""" SALIENCE """
class FeatureExtracter(nn.Module):
    def __init__(self):
        super( FeatureExtracter, self ).__init__() 
        self.conv1 = nn.Conv1d(3, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,dilation=1, return_indices=False,
                                 ceil_mode=False)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(16)
        
        self.conv3 = nn.Conv1d(16, 16, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm1d(16)

    def forward( self, x):
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
  
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        return x   
    
class LocalDiscriminator(nn.Module):
    def __init__(self, prob=0.5):
        super( LocalDiscriminator, self ).__init__()    
        self.prob = prob
        self.lstm1 = nn.LSTM(input_size=16, hidden_size=64, bidirectional=True, num_layers=1)
        self.out = nn.Linear(128, 2)
        
    def forward(self, x):
        x = x.permute(1, 0, 2) 
        
        x = F.dropout(x, p=self.prob, training=self.training) 
        x, (h_n,c_n) = self.lstm1(x)
           
        x = x[-1]
        
        x = F.dropout(x, p=self.prob, training=self.training)   
        x = self.out(x)
        return x      
    
    
class GlobalDiscriminator(nn.Module):
    def __init__(self, prob=0.5):
        super( GlobalDiscriminator, self ).__init__()    
        self.prob = prob
        self.lstm1 = nn.LSTM(input_size=16, hidden_size=64, bidirectional=True, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True, num_layers=1)
        self.out = nn.Linear(128, 2)
        
    def forward(self, x): 
        x = x.permute(1, 0, 2)
        
        x = F.dropout(x, p=self.prob, training=self.training) 
        x, (h_n,c_n) = self.lstm1(x)
        
        x = F.dropout(x, p=self.prob, training=self.training) 
        x, (h_n,c_n) = self.lstm2(x)
           
        x = x[-1]
        
        x = F.dropout(x, p=self.prob, training=self.training)   
        x = self.out(x)
        return x        
  

class ActivityClassifier(nn.Module):
    def __init__(self, prob=0.5):
        super( ActivityClassifier, self ).__init__()    
        self.prob = prob
        self.prob = prob
        self.lstm1 = nn.LSTM(input_size=16, hidden_size=64, bidirectional=True, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True, num_layers=1)
        self.out = nn.Linear(128, 12)
        
    def forward(self, x):
        x = x.permute(1, 0, 2)
        
        x = F.dropout(x, p=self.prob, training=self.training) 
        x, (h_n,c_n) = self.lstm1(x)
        
        x = F.dropout(x, p=self.prob, training=self.training) 
        x, (h_n,c_n) = self.lstm2(x)      
           
        x = x[-1]
        
        x = F.dropout(x, p=self.prob, training=self.training)   
        x = self.out(x)
        return x 
   
class AttentionNetwork(nn.Module):
    def __init__(self):
        super(AttentionNetwork, self ).__init__()    
        self.Q = nn.Linear(24, 24)
        self.K = nn.Linear(336, 24)
        
    def forward(self, local_features, concat_local_preds):  
        
        input_feature = torch.stack([local_features[i].reshape(-1, local_features[i].shape[1]*local_features[i].shape[2]) for i in range(12)], 1) 
        local_preds = concat_local_preds 
        
        query = self.Q(local_preds)
        key = self.K(input_feature)
        score = torch.matmul(key, query.reshape(-1, query.shape[1], 1)) 
        score = F.softmax(score / math.sqrt(query.size(-1)), dim = 1)
        score = score.reshape(-1, score.shape[1] * score.shape[2])
    
        return score 
    
