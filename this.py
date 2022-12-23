
#data = np.load('output_Abhay_Trikonasana.mp4.npy')
#print(data[10])

import numpy
import pandas as pd
import openpyxl

data_df = pd.read_excel('Output_keypoints.xlsx')
# data_df.insert(0,'Yoga Pose ',['Abhay_Bhujangasana',
# 'Abhay_Padmasana',
# 'Abhay_Shavasana',
# 'Abhay_Tadasana',
# 'Abhay_Trikonasana',
# 'Abhay_Vrikshasana',
# 'Ameya_Bhujangasana',
# 'Ameya_Padmasana',
# 'Ameya_Shavasana',
# 'Ameya_Tadasana',
# 'Ameya_Trikonasana',
# 'Ameya_Vrikshasana',
# 'Bhumi_Bhujangasana',
# 'Bhumi_Padmasana',
# 'Bhumi_Shavasana',
# 'Bhumi_Tadasana',
# 'Bhumi_Trikonasana',
# 'Bhumi_Vrikshasana',
# 'deepa_Bhujangasana',
# 'deepa_Padmasana',
# 'deepa_Shavasana',
# 'deepa_Tadasana',
# 'deepa_Trikonasana',
# 'deepa_Vrikshasana',
# 'Dristi_Bhujangasana',
# 'Dristi_Padmasana',
# 'Dristi_Shavasana',
# 'Dristi_Tadasana',
# 'Dristi_Trikonasana',
# 'Dristi_Vrikshasana',
# 'Harshav_Bhujangasana',
# 'Harshav_Padmasana',
# 'Harshav_Shavasana',
# 'Harshav_Tadasana',
# 'Harshav_Trikonasana',
# 'Harshav_Vrikshasana',
# 'Kaustuk_Bhujangasana',
# 'Kaustuk_Padmasana',
# 'Kaustuk_Shavasana',
# 'Kaustuk_Tadasana',
# 'Kaustuk_Trikonasana',
# 'Kaustuk_Vrikshasana',
# 'lakshmi_Bhujangasana',
# 'lakshmi_Padmasana',
# 'lakshmi_Shavasana',
# 'lakshmi_Tadasana',
# 'lakshmi_Vrikshasana',
# 'Piyush_Bhujangasana',
# 'Piyush_Padmasana',
# 'Piyush_Shavasana',
# 'Piyush_Tadasana',
# 'Piyush_Trikonasana',
# 'Piyush_Vrikshasana',
# 'Pranshul_Bhujangasana',
# 'Pranshul_Shavasana',
# 'Pranshul_Tadasana',
# 'Pranshul_Trikonasana',
# 'Pranshul_Vrikshasana',
# 'Rakesh_Bhujangasana',
# 'Rakesh_Padmasana',
# 'Rakesh_Shavasana',
# 'Rakesh_Tadasana',
# 'Rakesh_Trikonasana',
# 'Rakesh_Vrikshasana',
# 'Santosh2_Bhujangasana',
# 'Santosh_Bhujangasana',
# 'Santosh_Padmasana',
# 'Santosh_Shavasana',
# 'Santosh_Tadasana',
# 'Santosh_Trikonasana',
# 'Santosh_Vrikshasana',
# 'Sarthak_Bhujangasana',
# 'Sarthak_Padmasana',
# 'Sarthak_Tadasana',
# 'Sarthak_Trikonasana',
# 'Sarthak_Vrikshasana',
# 'Sathak_Shavasana',
# 'Shiva_Shavasana',
# 'Shiva_Tadasana',
# 'Shiva_Trikonasana',
# 'Shiva_Vrikshasana',
# 'Shiv_Bhujangasana',
# 'Shiv_Padmasana',
# 'Veena_Bhujangasana',
# 'veena_Padmasana',
# 'Veena_Shavasana',
# 'veena_Tadasana',
# 'veena_Vrikshasana'])
#data_df.insert(1,'Keypoints',[f'{data[]}'])
data = numpy.load('output_Abhay_Trikonasana.mp4.npy')
dt = numpy.array(data[310])
print(str(data[310]))
print(data_df)
#data[310] = data_df['Keypoints'].tolist()
data_df['Keypoints'][0] = str(data[310])
print(data_df)
data_df.to_excel(r'Output_keypoints.xlsx', index= False)
print('saved')


# import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(figsize=(4,4))

# ax = fig.add_subplot(111, projection='3d')
# fig = plt.figure(figsize=(4,4))


# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(data_df['x'],data_df['y'],data_df['z']) # plot the point (2,3,4) on the figure
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.show()

#excel = data_df.to_excel(r'\Keypoints.xlsx')