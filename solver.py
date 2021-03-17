# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:49:24 2020

@author: 82045
"""


from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from build_gen_opp import FeatureExtracter
from build_gen_opp import GlobalDiscriminator
from build_gen_opp import ActivityClassifier
from build_gen_opp import LocalDiscriminator
from build_gen_opp import AttentionNetwork
from dataset_read_opp import dataset_read
from tensorboardX import SummaryWriter
import sklearn.metrics as metrics

class Solver(object):
    def __init__(self, args):
        self.lr = args.lr
        self.seed = args.seed
        self.ex_num = args.ex_num
        self.test_user = args.test_user
        self.batch_size = args.batch_size

        print('dataset loading')
        self.datasets, self.dataset_test = dataset_read(self.batch_size, self.test_user)
        print('load finished!')
    
        self.FE = [ FeatureExtracter() for i in range(12) ]
        self.GD = GlobalDiscriminator()
        self.AC = ActivityClassifier()   
        self.LD = [ LocalDiscriminator() for i in range(12) ]
        self.WG = AttentionNetwork()

        for i in range(12):
            self.FE[i].cuda()
            self.LD[i].cuda()
        self.GD.cuda()
        self.AC.cuda()
        self.WG.cuda()

        self.set_optimizer(which_opt=args.optimizer, lr=args.lr)

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_fe = [ optim.SGD(self.FE[i].parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum) for i in range(12) ]
            self.opt_gd = optim.SGD(self.GD.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_ac = optim.SGD(self.AC.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_ld = [ optim.SGD(self.LD[i].parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum) for i in range(12) ]
            self.opt_wg = optim.SGD(self.WG.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)

        if which_opt == 'adam':
            self.opt_fe = [ optim.Adam(self.FE[i].parameters(),
                                   lr=lr, weight_decay=0.0005) for i in range(12) ]
            self.opt_gd = optim.Adam(self.GD.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_ac = optim.Adam(self.AC.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_ld = [ optim.Adam(self.LD[i].parameters(),
                                   lr=lr, weight_decay=0.0005) for i in range(12) ]
            self.opt_wg = optim.Adam(self.WG.parameters(),
                                     lr=lr, weight_decay=0.0005)            

    def reset_grad(self):
        for i in range(12):
            self.opt_fe[i].zero_grad()
            self.opt_ld[i].zero_grad()
        self.opt_gd.zero_grad()
        self.opt_ac.zero_grad()
        self.opt_wg.zero_grad()
  
    def train(self, maxepoch):
        save_path =  'ex' + str(self.ex_num) + '/' + 'user' + str(self.test_user) + '/'
        load_path = 'checkpoint/' + 'user' + str(self.test_user) + '/'
        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')
        os.mkdir(load_path)
        writer = SummaryWriter(save_path)
        
        torch.cuda.manual_seed(self.seed)
        criterion = nn.CrossEntropyLoss().cuda()
        
        for epoch in range(maxepoch):
            for i in range(12):
                self.FE[i].train()
                self.LD[i].train()
            self.GD.train()
            self.AC.train()
            self.WG.train()
            for batch_idx, data in enumerate(self.datasets):
                img_t = data['T']         #  img_t.size(128, 300, 113)
                img_s = data['S']         #  img_s.size(128, 300, 113)
                label_s = data['S_label'] #  label_s.size(128, ) 
                label_s_st = torch.zeros(label_s.size()[0])
                label_t_st = torch.ones(label_s.size()[0])

                if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                    break
                img_s = img_s.cuda()
                img_t = img_t.cuda()
                label_s = Variable(label_s.long().cuda())
                label_s_st = Variable(label_s_st.long().cuda())
                label_t_st = Variable(label_t_st.long().cuda())

                img_s = Variable(img_s)     # (128, 300, 113)
                img_t = Variable(img_t)     # (128, 300, 113)
                self.reset_grad()
                
                section = [3,3,3,3,3,3, 3,3,3,3,3,3]
                
                img_s_split = torch.split(img_s, section, dim=2)
                img_s_list = []
                for i in range(12):
                    temp = img_s_split[i]
                    if temp.shape[2] == 1:
                        temp = temp.repeat(1,1,3)
                    img_s_list.append(temp)
                    
                img_t_split = torch.split(img_t, section, dim=2)
                img_t_list = []
                for i in range(12):
                    temp = img_t_split[i]
                    if temp.shape[2] == 1:
                        temp = temp.repeat(1,1,3)
                    img_t_list.append(temp)                    
                        
                """ step 1 """
                local_features = [ self.FE[i](img_s_list[i]) for i in range(12) ]
                local_preds = [ self.LD[i](local_features[i]) for i in range(12) ]
                concat_local_preds = torch.cat( [local_preds[i] for i in range(12)], 1) 
                weights = self.WG(local_features, concat_local_preds)
                weighted_features = torch.zeros(local_features[0].shape[0], local_features[0].shape[1], local_features[0].shape[2]).cuda()
                for i in range(12):
                    weighted_features.add_(local_features[i])
                    weighted_features.add_(torch.mul(local_features[i], weights[:, i].reshape(weights[:, 1].shape[0], 1, 1)))
                feat_s = weighted_features
                output_s1 = self.AC(feat_s)
                loss_s = criterion(output_s1, label_s) 
                loss_s.backward()
                for i in range(12):
                    self.opt_fe[i].step()
                self.opt_wg.step()
                self.opt_ac.step()
                self.reset_grad()
                
                """ step 2 """
                local_features = [ self.FE[i](img_s_list[i]) for i in range(12) ]         
                local_preds = [ self.LD[i](local_features[i]) for i in range(12) ]        
                concat_local_preds = torch.cat( [local_preds[i] for i in range(12)], 1) 
                weights = self.WG(local_features, concat_local_preds)
                weighted_features = torch.zeros(local_features[0].shape[0], local_features[0].shape[1], local_features[0].shape[2]).cuda()
                for i in range(12):
                    weighted_features.add_(local_features[i])
                    weighted_features.add_(torch.mul(local_features[i], weights[:, i].reshape(weights[:, 1].shape[0], 1, 1)))
                feat_s = weighted_features
                output_s1 = self.AC(feat_s)
                output_sd = self.GD(feat_s)
                loss_ld_s = [ criterion(local_preds[i], label_s_st) for i in range(12) ]
                loss_ld_s = torch.mean( torch.stack( [loss_ld_s[i] for i in range(12)], 0), dim=0)
                
                local_features = [ self.FE[i](img_t_list[i]) for i in range(12) ]         
                local_preds = [ self.LD[i](local_features[i]) for i in range(12) ]        
                concat_local_preds = torch.cat( [local_preds[i] for i in range(12)], 1) 
                weights = self.WG(local_features, concat_local_preds)
                weighted_features = torch.zeros(local_features[0].shape[0], local_features[0].shape[1], local_features[0].shape[2]).cuda()
                for i in range(12):
                    weighted_features.add_(local_features[i])
                    weighted_features.add_(torch.mul(local_features[i], weights[:, i].reshape(weights[:, 1].shape[0], 1, 1)))
                feat_t = weighted_features  
                output_td = self.GD(feat_t)
                loss_ld_t = [ criterion(local_preds[i], label_t_st) for i in range(12) ]
                loss_ld_t = torch.mean( torch.stack( [loss_ld_t[i] for i in range(12)], 0), dim=0)
                
                loss_s1 = criterion(output_s1, label_s)
                loss_sd = criterion(output_sd, label_s_st)
                loss_td = criterion(output_td, label_t_st)
                loss_ld = loss_ld_s + loss_ld_t
                loss_gd = loss_sd + loss_td
                loss_dis = 0.5 * loss_ld + 0.5 * loss_gd
                loss = loss_dis
                loss.backward()
                self.opt_gd.step()
                for i in range(12):
                    self.opt_ld[i].step()
                self.reset_grad()
                
                """ step 3 """
                local_features = [ self.FE[i](img_s_list[i]) for i in range(12) ]         
                local_preds = [ self.LD[i](local_features[i]) for i in range(12) ]        
                concat_local_preds = torch.cat( [local_preds[i] for i in range(12)], 1) 
                weights = self.WG(local_features, concat_local_preds)
                weighted_features = torch.zeros(local_features[0].shape[0], local_features[0].shape[1], local_features[0].shape[2]).cuda()
                for i in range(12):
                    weighted_features.add_(local_features[i])
                    weighted_features.add_(torch.mul(local_features[i], weights[:, i].reshape(weights[:, 1].shape[0], 1, 1)))
                feat_s = weighted_features
                output_s1 = self.AC(feat_s)
                output_sd = self.GD(feat_s)
                loss_ld_s = [ criterion(local_preds[i], label_s_st) for i in range(12) ]
                loss_ld_s = torch.mean( torch.stack( [loss_ld_s[i] for i in range(12)], 0), dim=0)
                    
                local_features = [ self.FE[i](img_t_list[i]) for i in range(12) ]         
                local_preds = [ self.LD[i](local_features[i]) for i in range(12) ]        
                concat_local_preds = torch.cat( [local_preds[i] for i in range(12)], 1) 
                weights = self.WG(local_features, concat_local_preds)
                weighted_features = torch.zeros(local_features[0].shape[0], local_features[0].shape[1], local_features[0].shape[2]).cuda()
                for i in range(12):
                    weighted_features.add_(local_features[i])
                    weighted_features.add_(torch.mul(local_features[i], weights[:, i].reshape(weights[:, 1].shape[0], 1, 1)))
                feat_t = weighted_features  
                output_td = self.GD(feat_t)
                loss_ld_t = [ criterion(local_preds[i], label_t_st) for i in range(12) ]
                loss_ld_t = torch.mean( torch.stack( [loss_ld_t[i] for i in range(12)], 0), dim=0)
                    
                loss_s1 = criterion(output_s1, label_s)
                loss_sd = criterion(output_sd, label_s_st)
                loss_td = criterion(output_td, label_t_st)
                loss_ld = loss_ld_s + loss_ld_t
                loss_gd = loss_sd + loss_td
                loss_dis = 0.5 * loss_ld + 0.5 * loss_gd
                loss = -loss_dis
                loss.backward()
                for i in range(12):
                    self.opt_fe[i].step()
                self.opt_wg.step()
                self.reset_grad()
            
            """ test """
            for i in range(12):
                self.FE[i].eval()
                self.LD[i].eval()
            self.GD.eval()
            self.AC.eval()
            self.WG.eval()
            test_loss = 0
            correct1 = 0
            size = 0
            predicts = []
            labelss = []
            for batch_idx, data in enumerate(self.dataset_test):
                img = data['T']
                label = data['T_label']
                label_t_st = torch.ones(label_s.size()[0])
                
                img, label, label_t_st = img.cuda(), label.long().cuda(), label_t_st.long().cuda()
                img, label, label_t_st = Variable(img, volatile=True), Variable(label), Variable(label_t_st)
                
                section = [3,3,3,3,3,3, 3,3,3,3,3,3]
                img_t_split = torch.split(img, section, dim=2)
                img_t_list = []
                for i in range(12):
                    temp = img_t_split[i]
                    if temp.shape[2] == 1:
                        temp = temp.repeat(1,1,3)
                    img_t_list.append(temp)
                    
                local_features = [ self.FE[i](img_t_list[i]) for i in range(12) ]         
                local_preds = [ self.LD[i](local_features[i]) for i in range(12) ]        
                concat_local_preds = torch.cat( [local_preds[i] for i in range(12)], 1) 
                weights = self.WG(local_features, concat_local_preds)                
                weighted_features = torch.zeros(local_features[0].shape[0], local_features[0].shape[1], local_features[0].shape[2]).cuda()
                for i in range(12):
                    weighted_features.add_(local_features[i])
                    weighted_features.add_(torch.mul(local_features[i], weights[:, i].reshape(weights[:, 1].shape[0], 1, 1)))
                feat_t = weighted_features 
                output1 = self.AC(feat_t)
                test_loss += F.nll_loss(F.log_softmax(output1,dim=1), label, reduction='sum').item()
                pred1 = output1.data.max(1)[1] 
                k = label.data.size()[0] 
                correct1 += pred1.eq(label.data).cpu().sum()
                size += k
                
                labels = label.cpu().numpy()
                labelss = labelss + list(labels)
                pred1s = pred1.cpu().numpy()
                predicts = predicts + list(pred1s)
                
            test_loss = test_loss / size
            
            f1 = metrics.f1_score(labelss, predicts, average='macro')
            writer.add_scalar('loss1', loss_s1.item(), global_step=epoch)
            writer.add_scalar('discrepancy', loss_dis.item(), global_step=epoch)
            writer.add_scalar('testloss', test_loss, global_step=epoch)
            writer.add_scalar('testacc', 1. * correct1.float() / size, global_step=epoch)
            writer.add_scalar('testf1', f1, global_step=epoch)
            
            print('Train Epoch:{} Loss1:{:.6f} Testloss:{:.6f} Testacc:{:.6f} Testf1:{:.6f}'.format(
                epoch, loss_s1.item(), test_loss, 1. * correct1.float() / size, f1)) 