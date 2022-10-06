'''
download nmnist dataset from https://www.garrickorchard.com/datasets/n-mnist
in this project, the data and label for training and test dataset are stored seperately
'''

import torch
import torch.utils.data as data

class GetData(data.Dataset):
    def __init__(self, path_data, path_label):
        self.data = torch.load(path_data)
        self.label = torch.load(path_label)
        self.num_sample = int((len(self.data) // 100) * 100)

    def __getitem__(self, index):
        data, target = self.data[index], self.label[index]
        return data, target

    def __len__(self):
        return self.num_sample