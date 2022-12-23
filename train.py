from random import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from model import siamies
from torch.utils.data import Dataset,DataLoader
import random
from numpi import yoga
class traniner():
    def __init__(self):
        self.model_obj = siamies()
        self.data_obj = yoga()
        self.loss_obj = nn.TripletMarginLoss(margin=2.2)
        self.optimizer_obj = optim.SGD(params=self.model_obj.parameters(),lr=0.02)

    def train(self):
        n_epochs = 80
        anchor_list = []
        positive_list = []
        negative_list = []


        for ind in range(50):
            rand_int = random.randint(0, len(self.data_obj.asanas)-1)
            temp_anchor, temp_positive, temp_negative = self.data_obj[self.data_obj.asanas[rand_int]]
            anchor_list.append(temp_anchor)
            positive_list.append(temp_positive)
            negative_list.append(temp_negative)

        for i in range(n_epochs):
            for j in range(50):
                temp_anc = anchor_list[j]
                temp_pos = positive_list[j]
                temp_neg = negative_list[j]

                temp_anc = temp_anc[None,None,:,:]
                temp_pos = temp_pos[None,None,:,:]
                temp_neg = temp_neg[None,None,:,:]
                print(type(temp_anc))
                print(temp_anc.shape)
                
                emb_anc = self.model_obj(temp_anc)
                emb_pos = self.model_obj(temp_pos)
                emb_neg = self.model_obj(temp_neg)

                loss = self.loss_obj(emb_anc, emb_pos, emb_neg)
                self.optimizer_obj.zero_grad()
                loss.backward()
                self.optimizer_obj.step()
                print(f'loss is {loss} at epoch {i} at example {j}')


train_obj = traniner()
#train_obj.train()

# a,_,_ = train_obj.data_obj['Bhujangasana']
# a = a[None,None,:,:]
# a_emb = train_obj.model_obj(a)

# b,_,_ = train_obj.data_obj['Bhujangasana']
# b = b[None,None,:,:]
# b_emb = train_obj.model_obj(b)

# c,_,_ = train_obj.data_obj['Padmasana']
# c = c[None,None,:,:]
# c_emb = train_obj.model_obj(c)

# d,_,_ = train_obj.data_obj['Padmasana']
# d = d[None,None,:,:]
# d_emb = train_obj.model_obj(d)

# #print(a_emb, b_emb, c_emb, d_emb)

# print(a_emb-b_emb)
# print(c_emb-d_emb)
# print(a_emb-c_emb)
# print(b_emb-d_emb)

# inp = input('Do you want to save the model? (Y/N)')
# if inp == 'Y' or inp =='y':
#     p1 = 'seimies_model.pth'
#     p2 = 'seimies_state_dict.pth'
#     torch.save(train_obj.model_obj,p1 )
#     torch.save(train_obj.model_obj.state_dict(),p2)
#     print('Model Saved')

inp2 = input('Do you want to save the embeddings? (Y/N)')
if inp2 =="Y" or inp2 =="y":
    train_obj.data_obj.data_embed_save()