import os
import pickle


data_dict = {}
train_images = os.listdir('./COD10K-v3/Train/Image')
test_images = os.listdir('./COD10K-v3/Test/Image')

train_images = [x.removesuffix('.jpg') for x in train_images]
data_dict['train'] = {}
data_dict['train']['name_list'] = train_images
test_images = [x.removesuffix('.jpg') for x in test_images]
data_dict['test'] = {}
data_dict['test']['name_list'] = test_images

sav_path = 'cod10k_pickle.pkl'
with open(sav_path, 'wb') as file:
    pickle.dump(data_dict, file)
