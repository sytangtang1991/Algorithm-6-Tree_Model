#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:31:17 2020

@author: yangsong
"""

# !pip install numpy==1.18
# !pip install scipy==1.1.0
# !pip install pandas==0.25.0
# !pip install pandas --upgrade --ignore-installed pyparsing
!pip install pandas --upgrade
# !pip install pip==20.1

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import math;
from math import log;
###########################
# Part 1: Create dataset
###########################
datasets = [['Young', 'No', 'No', 'Moderate', 'No'],
            ['Young', 'No', 'No', 'Good', 'No'],
            ['Young', 'Yes', 'No', 'Good', 'Yes'],
            ['Young', 'Yes', 'Yes', 'Moderate', 'Yes'],
            ['Young', 'No', 'No', 'Moderate', 'No'],
            ['Middle', 'No', 'No', 'Moderate', 'No'],
            ['Middle', 'No', 'No', 'Good', 'No'],
            ['Middle', 'Yes', 'Yes', 'Good', 'Yes'],
            ['Middle', 'No', 'Yes', 'Excellent', 'Yes'],
            ['Middle', 'No', 'Yes', 'Excellent', 'Yes'],
            ['Old', 'No', 'Yes', 'Excellent', 'Yes'],
            ['Old', 'No', 'Yes', 'Good', 'Yes'],
            ['Old', 'Yes', 'No', 'Good', 'Yes'],
            ['Old', 'Yes', 'No', 'Excellent', 'Yes'],
            ['Old', 'No', 'No', 'Moderate', 'No'],
            ]
datasets = pd.DataFrame(datasets,columns=['Age','Work','Own_Apartment','Credit_History','Y'])

# Count levels by each variable
print(pd.crosstab(datasets['Age'],columns='count'));
print(pd.crosstab(datasets['Work'],columns='count'));
print(pd.crosstab(datasets['Own_Apartment'],columns='count'));
print(pd.crosstab(datasets['Credit_History'],columns='count'));
print(pd.crosstab(datasets['Y'],columns='count'));

##########################
# Part 2: Information Gain
##########################
# Part 2.1： Calculate Entropy
def get_entropy(datasets,response):
    y_unique = list(set(datasets[response]))
    # Calculate Prior Probability
    prior_prob = np.zeros(len(y_unique))
    y_level = len(y_unique)
    
    for i in range(0,len(y_unique)):
        prior_prob[i]=sum(datasets[response]==y_unique[i])/len(datasets[response])
    # Calculate Entropy
    entropy=0
    for i in range(0,len(y_unique)):
        temp = -1 * prior_prob[i]*log(prior_prob[i],len(y_unique))
        entropy += temp
    return entropy,y_unique
# Test
entropy_test,y_unique = get_entropy(datasets,'Y')

# Part 2.2: Calculate Conditional Entropy
def get_conditional_entropy(datasets,col,response):
    var=list(datasets.columns)[col];
    var_unique = list(set(datasets[var]));
    y_unique = list(set(datasets[response]));
    y_level = len(y_unique)
    # calculate conditional probability
    conditional_prob = np.zeros([len(var_unique),y_level+1])
    
    for i in range(0,len(var_unique)):
        conditional_prob[i,0]=sum(datasets[var]==var_unique[i])/len(datasets[var])
    
    for i in range(0,len(var_unique)):
        for j in range(0,len(y_unique)):
            conditional_prob[i,j+1]=datasets.loc[(datasets[var]==var_unique[i])&(datasets[response]==y_unique[j]),].shape[0]/sum(datasets[var]==var_unique[i])
    
    # calculate conditional entropy
    cond_entropy=0
    for i in range(0,len(var_unique)):
        if (conditional_prob[i,1]!=0) & (conditional_prob[i,2]!=0):
            temp = -1 * conditional_prob[i,0]*(conditional_prob[i,1]*log(conditional_prob[i,1],len(y_unique))+conditional_prob[i,2]*log(conditional_prob[i,2],len(y_unique)))
        elif (conditional_prob[i,1]==0) & (conditional_prob[i,2]!=0):
            temp = -1 * conditional_prob[i,0]*(conditional_prob[i,2]*log(conditional_prob[i,2],len(y_unique)))
        elif (conditional_prob[i,1]!=0) & (conditional_prob[i,2]==0):
            temp = -1 * conditional_prob[i,0]*(conditional_prob[i,1]*log(conditional_prob[i,1],len(y_unique)))
            
        cond_entropy += temp

    return cond_entropy

# Test
cond_entropy_test1=get_conditional_entropy(datasets=datasets,col=0,response='Y')
cond_entropy_test2=get_conditional_entropy(datasets=datasets,col=1,response='Y')
cond_entropy_test3=get_conditional_entropy(datasets=datasets,col=2,response='Y')
cond_entropy_test4=get_conditional_entropy(datasets=datasets,col=3,response='Y')

# Part 2.3: Information Gains
def info_gain(entropy,cond_entropy):
    info_gain=entropy-cond_entropy
    return info_gain
# Test
info_gain_test1=info_gain(entropy_test,cond_entropy_test1)
info_gain_test2=info_gain(entropy_test,cond_entropy_test2)
info_gain_test3=info_gain(entropy_test,cond_entropy_test3)
info_gain_test4=info_gain(entropy_test,cond_entropy_test4)    


def info_gain_train(datasets):
    # Calculate Entropy
    entropy,y_unique=get_entropy(datasets,'Y')
    
    best_feature=[]
    # Calculate Conditional Entropy
    for i in range(0,(datasets.shape[1]-1)):
        var=list(datasets.columns)[i];
        var_cond_entropy=get_conditional_entropy(datasets,i,'Y');
        var_info_gain=info_gain(entropy,var_cond_entropy)
        
        best_feature.append((var,var_info_gain))
    
    best_ = max(best_feature, key=lambda x: x[-1])
    
    return best_

# Test
best_feature_test,best_test=info_gain_train(datasets)
print(best_feature_test)
print(best_test)


# Binary Tree Model
class Node:
    def __init__ (self,root=True,label=None, feature_name=None, feature=None):
        self.root=root
        self.label=label
        self.feature_name=feature_name
        self.feature=feature
        self.tree = {}
        self.result={
                'label':self.label,
                'feature':self.feature,
                'tree':self.tree
                }
    
    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self,val,node):
        self.tree[val] = node
    
    def predict(self,features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)



class Dtree:
    def __init__ (self,epsilon=0.1):
        self.epsilon=0.1
        self._tree = {}
        
    # Part 1： Calculate Entropy
    def get_entropy(self,datasets,response):
        y_unique = list(set(datasets[response]))
        # Calculate Prior Probability
        prior_prob = np.zeros(len(y_unique))
        y_level = len(y_unique)
    
        for i in range(0,len(y_unique)):
            prior_prob[i]=sum(datasets[response]==y_unique[i])/len(datasets[response])
        # Calculate Entropy
        entropy=0
        for i in range(0,len(y_unique)):
            temp = -1 * prior_prob[i]*log(prior_prob[i],len(y_unique))
            entropy += temp
        return entropy,y_unique
    
    # Part 2: Calculate Conditional Entropy
    def get_conditional_entropy(self,datasets,col,response):
        var=list(datasets.columns)[col]
        var_unique = list(set(datasets[var]))
        y_unique = list(set(datasets[response]))
        y_level=len(y_unique)
        # calculate conditional probability
        conditional_prob = np.zeros([len(var_unique),y_level+1])
    
        for i in range(0,len(var_unique)):
            conditional_prob[i,0]=sum(datasets[var]==var_unique[i])/len(datasets[var])
    
        for i in range(0,len(var_unique)):
            for j in range(0,len(y_unique)):
                conditional_prob[i,j+1]=datasets.loc[(datasets[var]==var_unique[i])&(datasets[response]==y_unique[j]),]\
                                .shape[0]/sum(datasets[var]==var_unique[i])
    
        # calculate conditional entropy
        cond_entropy=0
        for i in range(0,len(var_unique)):
            if (conditional_prob[i,1]!=0) & (conditional_prob[i,2]!=0):
                temp = -1 * conditional_prob[i,0]*(conditional_prob[i,1]*log(conditional_prob[i,1],len(y_unique))+\
                                            conditional_prob[i,2]*log(conditional_prob[i,2],len(y_unique)))
            elif (conditional_prob[i,1]==0) & (conditional_prob[i,2]!=0):
                temp = -1 * conditional_prob[i,0]*(conditional_prob[i,2]*log(conditional_prob[i,2],len(y_unique)))
            elif (conditional_prob[i,1]!=0) & (conditional_prob[i,2]==0):
                temp = -1 * conditional_prob[i,0]*(conditional_prob[i,1]*log(conditional_prob[i,1],len(y_unique)))
            
            cond_entropy += temp

        return cond_entropy

    # Part 3: Information Gains
    def info_gain(self,entropy,cond_entropy):
        info_gain=entropy-cond_entropy
        return info_gain

    # Part 4: Train Model
    def info_gain_train(self,datasets):
        # Calculate Entropy
        entropy,y_unique=self.get_entropy(datasets,'Y')
    
        best_feature=[]
        # Calculate Conditional Entropy
        for i in range(0,(datasets.shape[1]-1)):
            var=list(datasets.columns)[i];
            var_cond_entropy=self.get_conditional_entropy(datasets,i,'Y');
            var_info_gain=self.info_gain(entropy,var_cond_entropy)
        
            best_feature.append((var,var_info_gain))
        
        best_ = max(best_feature, key=lambda x: x[-1])
         
        return best_

    def train(self,train_data):
        
        _, y_train, features = train_data.iloc[:,:-1],train_data.iloc[:,-1],train_data.columns[:-1]
        
        # Senario 1: If all response is same, then the tree model has single node. Return T
        if len(y_train.value_counts())==1:
            return Node(root=True,
                        label=y_train.iloc[0])
        
        # Senario 2: If features are none, then the tree model has single node. Return T using model of response
        if len(features)==0:
            return Node(root=True,
                        label=y_train.value_counts().sort_values(ascending=False).index[0])
        
        # Senario 3:
        max_feature,max_info_gain=self.info_gain_train(train_data);
        max_feature_name = max_feature
        
        # Senario 4:
        if max_info_gain<self.epsilon:
            return Node(root=True,
                        label=y_train.value_counts().sort_values(ascending=False).index[0])
        
        # Senario 5:
        node_tree = Node(root=False,
                         feature_name = max_feature_name,
                         feature = max_feature)
        feature_list = train_data[max_feature_name].value_counts().index
                                 
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)
        
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f,sub_tree)
        
        return node_tree
        
    def fit(self,train_data):
        self._tree = self.train(train_data)
        return self._tree
    
    def predict(self,X_test):
        return self._tree.predict(X_test)

# Run Tree Model
dt = Dtree()
tree = dt.fit(datasets)
print(tree)



