from random import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from model import siamies
from torch.utils.data import Dataset,DataLoader
import random
from model import siamies


class yoga(Dataset):
    def __init__(self):
        df = pd.read_csv('time.csv')

        self.asan_keypoints = {'Padmasana':[], 'Shavasana':[], 'Tadasana':[], 'Trikonasana':[], 'Vrikshasana':[], 'Bhujangasana':[]}
        self.people_list = ['Abhay', 'Ameya', 'Bhumi', 'deepa', 'Dristi', 'Harshav', 'Kaustuk', 'lakshmi', 'Piyush', 'Pranshul',
         'Rakesh', 'Santosh', 'Sarthak', 'Shiva', 'Veena']
        self.asanas= ['Padmasana', 'Shavasana', 'Tadasana', 'Trikonasana', 'Vrikshasana', 'Bhujangasana']

        df_asan = df['Asan']
        df_timestamp = df['timestamp']
        df_timestamp.index = df_asan

        for i in self.people_list:
            for j in self.asanas:
                try:
                    loaded_array = np.load(i +'_'+j+'.npy')
                    
                    timestamp = df_timestamp[i+'_'+j]
                    ind = timestamp * 30 + 10
                    loaded_array = loaded_array[ind]
                    loaded_array = np.array([loaded_array[10], loaded_array[9], loaded_array[8],
                                            loaded_array[11], loaded_array[8], loaded_array[14],
                                            loaded_array[11], loaded_array[12], loaded_array[13],
                                            loaded_array[14], loaded_array[15], loaded_array[16],
                                            loaded_array[8], loaded_array[7], loaded_array[0],
                                            loaded_array[4], loaded_array[0], loaded_array[1],
                                            loaded_array[4], loaded_array[5], loaded_array[6],
                                            loaded_array[1], loaded_array[2], loaded_array[3]])
                    #print(loaded_array.shape)
                    self.asan_keypoints[j].append(torch.from_numpy(loaded_array))
                except FileNotFoundError:
                    pass
        #print(self.asan_keypoints['Padmasana'][0].shape)
        count = 0
        for aasan in self.asanas:
            count+=len(self.asan_keypoints[aasan])
            #print(len(self.asan_keypoints[aasan]))
        #print(count)

    def __getitem__(self, index):
        rand_int1 = random.randint(0,len(self.asan_keypoints[index])-1)
        #print(rand_int1)
        anchor = self.asan_keypoints[index][rand_int1]

        rand_int2 = random.randint(0,len(self.asan_keypoints[index])-1)
        #print(rand_int2)
        while rand_int1==rand_int2:
            rand_int2 = random.randint(0,len(self.asan_keypoints[index])-1)
            #print(rand_int2)

        positive = self.asan_keypoints[index][rand_int2]
        rand_int3 = random.randint(0,len(self.asanas)-1)
        #print(rand_int3)
        while self.asanas[rand_int3] == index:
            rand_int3 = random.randint(0,len(self.asanas)-1)
            #print(rand_int3)

        rand_int4 = random.randint(0,len(self.asan_keypoints[self.asanas[rand_int3]])-1)
        #print(rand_int4)
        negative = self.asan_keypoints[self.asanas[rand_int3]][rand_int4]
        return anchor, positive, negative

    def data_embed_save(self):
        for i in range(len(self.asanas)):
            for j in range(len(self.asan_keypoints[self.asanas[i]])):
                temp = self.asan_keypoints[self.asanas[i]][j]
                temp = temp[None,None,:,:]
                model_obj = siamies()
                model_obj.load_state_dict(torch.load('seimies_state_dict.pth'))
                model_obj = model_obj.eval()
                temp_embed = model_obj(temp)
                
                path = self.asanas[i]+'_'+str(j)+'.pt'
                torch.save(temp_embed,path)
    

    def transform(self, loaded_array, timestamp):

        ind = timestamp * 30 + 10
        loaded_array = loaded_array[ind]

        loaded_array = np.array([loaded_array[10], loaded_array[9], loaded_array[8],
                                            loaded_array[11], loaded_array[8], loaded_array[14],
                                            loaded_array[11], loaded_array[12], loaded_array[13],
                                            loaded_array[14], loaded_array[15], loaded_array[16],
                                            loaded_array[8], loaded_array[7], loaded_array[0],
                                            loaded_array[4], loaded_array[0], loaded_array[1],
                                            loaded_array[4], loaded_array[5], loaded_array[6],
                                            loaded_array[1], loaded_array[2], loaded_array[3]])
        
        
        return torch.from_numpy(loaded_array)



#y1 = yoga()
#a,b,c = y1['Padmasana']
#print(a.shape, b.shape, c.shape)
# temp = asan_keypoints['Padmasana'][0]
# temp = temp.reshape(1,1,24,3)
# temp = torch.from_numpy(temp)

# model_obj = siamies()

# temp_model = model_obj(temp)
# print(temp_model.shape)