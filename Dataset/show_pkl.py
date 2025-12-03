import pickle
import os

pkl_paths = [
    'cod/cod10k_pickle.pkl',
]

for task_pkl in pkl_paths:
    with open(task_pkl, 'rb') as file:
        loaded_dict = pickle.load(file)

    train_name_list = loaded_dict['train']['name_list']
    test_name_list  = loaded_dict['test']['name_list']

    print(train_name_list)
    print(test_name_list)
    print('train num: {}'.format(len(train_name_list)))
    print('test num: {}'.format(len(test_name_list)))