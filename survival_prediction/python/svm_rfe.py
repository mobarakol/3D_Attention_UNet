#System
import numpy as np
import sys
import os
import random
from glob import glob
from skimage import io
from PIL import Image
import random
import SimpleITK as sitk
#Torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function
import torch
import torch.nn as nn
import torchvision.transforms as standard_transforms
#from torchvision.models import resnet18
import nibabel as nib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_friedman1
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.svm import SVC

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
ckpt_path = 'ckpt'
exp_name = 'lol'
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists(os.path.join(ckpt_path, exp_name)):
    os.makedirs(os.path.join(ckpt_path, exp_name))
args = {
    'num_class': 2,
    'num_gpus': 1,
    'start_epoch': 1,
    'num_epoch': 100,
    'batch_size': 1,
    'lr': 0.01,
    'lr_decay': 0.9,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',
    'opt': 'adam',
    'crop_size1': 138,

}

class HEMDataset(Dataset):
    def __init__(self, text_dir):
        file_pairs = open(text_dir,'r')
        self.img_anno_pairs = file_pairs.readlines()
        self.req_file, self.req_tar = [],[]
        for i in range(len(self.img_anno_pairs)):
            net = self.img_anno_pairs[i][:-1]
            self.req_file.append(net[:3])
            self.req_tar.append(net[4])


    def __len__(self):
        return len(self.req_tar)

    def __getitem__(self, index):
        _file_num = self.req_file[index]
        _gt = float(self.req_tar[index])

        req_npy = './Features_Train/'+ str(_file_num) + 'ct1_seg.npy'
        _input_arr = np.load(req_npy, allow_pickle=True)
        _input = np.array([])
        for i in range(len(_input_arr)):
            _input = np.concatenate((_input, _input_arr[i]), axis=None)
        _input = torch.from_numpy(np.array(_input)).float()
        _target = torch.from_numpy(np.array(_gt)).long()

        return _input, _target

class HEMDataset_test(Dataset):
    def __init__(self, text_dir):
        file_pairs = open(text_dir,'r')
        self.img_anno_pairs = file_pairs.readlines()
        self.req_file, self.req_tar = [],[]
        for i in range(len(self.img_anno_pairs)):
            net = self.img_anno_pairs[i][:-1]
            self.req_file.append(net[:3])
            self.req_tar.append(net[4])


    def __len__(self):
        return len(self.req_tar)

    def __getitem__(self, index):
        _file_num = self.req_file[index]
        _gt = float(self.req_tar[index])

        req_npy = './Features_Val/'+ str(_file_num) + 'ct1_seg.npy'
        _input_arr = np.load(req_npy, allow_pickle=True)
        _input = np.array([])
        for i in range(len(_input_arr)):
            _input = np.concatenate((_input, _input_arr[i]), axis=None)
        #print(_input)
        _input = torch.from_numpy(np.array(_input)).float()
        _target = torch.from_numpy(np.array(_gt)).long()

        return _input, _target

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def important_features(fea, idx):
    batch = []
    for j in range(len(fea)):
        req_inputs = []
        for i in idx[0]:
            req_inputs.append(fea[0][i])
        batch.append(req_inputs)
    return req_inputs

if __name__ == '__main__':

    train_file = 'Train_dir.txt'
    test_file = 'Val_dir.txt'
    train_dataset = HEMDataset(text_dir=train_file)
    test_dataset = HEMDataset_test(text_dir=test_file)
    rfe_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2,drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2,drop_last=False)

    max_epoch = 1
    X_rfe, Y_rfe = [], []
    for epoch in range (max_epoch):
        for batch_idx, data in enumerate(rfe_loader):
            inputs, labels = data
            inputs, labels = inputs.cpu().numpy(), labels.cpu().numpy()
            X_rfe.append(inputs)
            Y_rfe.append(labels)

    X_rfe, Y_rfe = np.squeeze(X_rfe, axis=1), np.squeeze(Y_rfe, axis=1)
    rfe_model = SVR(kernel="linear")
    rfe = RFE(rfe_model, 5, step=1)
    fit = rfe.fit(X_rfe, Y_rfe)

    rank = fit.ranking_
    req_idx = np.where(rank == 1)
    print(fit.ranking_)
    print('Finished RFE')

    X_train, Y_train = [], []
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        req_inputs = important_features(inputs.cpu().numpy(), req_idx)
        X_train.append(req_inputs)
        Y_train.append(labels.cpu().numpy())
    X_train, Y_train = np.array(X_train), np.squeeze(np.array(Y_train), axis=1)

    X_test, Y_test = [], []
    for batch_idx, data in enumerate(test_loader):
        inputs, labels = data
        req_inputs = important_features(inputs.cpu().numpy(), req_idx)
        X_test.append(req_inputs)
        Y_test.append(labels.cpu().numpy())
    X_test, Y_test = np.array(X_test), np.squeeze(np.array(Y_test), axis=1)

    score, count = [], []
    #model = LogisticRegression()
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    score.append(sum(model.predict(X_test) == Y_test))
    count.append(len(Y_test))
    print(model.predict(X_test))
    print(score)

    #print(X_train.shape, Y_train.shape, X_test.shape,Y_test.shape)