# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:18:07 2020

@author: 82045
"""
import model_opp

def FeatureExtracter():
    return model_opp.FeatureExtracter()

def GlobalDiscriminator():
    return model_opp.GlobalDiscriminator()    

def ActivityClassifier():    
    return model_opp.ActivityClassifier()  

def LocalDiscriminator():
    return model_opp.LocalDiscriminator()

def AttentionNetwork():
    return model_opp.AttentionNetwork()

     
    