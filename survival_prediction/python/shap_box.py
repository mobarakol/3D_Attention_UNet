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
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_friedman1
from sklearn.svm import LinearSVC
from sklearn.svm import SVR

import xgboost
import shap
import sklearn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    'lr': 3,
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
        print(self.img_anno_pairs)
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
        print(self.img_anno_pairs)
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
        _input = torch.from_numpy(np.array(_input)).float()
        _target = torch.from_numpy(np.array(_gt)).long()

        return _input, _target

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 2)
        self.out_act = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.out_act(x)
        return x



if __name__ == '__main__':

    train_file = 'Train_dir.txt'
    test_file = 'Val_dir.txt'
    train_dataset = HEMDataset(text_dir=train_file)
    test_dataset = HEMDataset_test(text_dir=test_file)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2,drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2,drop_last=False)

    max_epoch = 1
    X_train, Y_train = [], []
    for epoch in range (max_epoch):
        for batch_idx, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.cpu().numpy(), labels.cpu().numpy()
            X_train.append(inputs)
            Y_train.append(labels)
            print(batch_idx)
    print('okay')
    X_train, Y_train = np.squeeze(X_train, axis=1), np.squeeze(Y_train, axis=1)
    print(X_train.shape, Y_train.shape)

    X_test, Y_test = [], []
    for epoch in range(max_epoch):
        for batch_idx, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.cpu().numpy(), labels.cpu().numpy()
            X_test.append(inputs)
            Y_test.append(labels)
    print('okay')
    X_test, Y_test = np.squeeze(X_test, axis=1), np.squeeze(Y_test, axis=1)
    print(X_test.shape, Y_test.shape)

    X_train, X_test, Y_train, Y_test = np.array(X_train, dtype='f'), np.array(X_test, dtype='f'), np.array(Y_train, dtype='f'), np.array(Y_test, dtype='f')
    print(np.max(X_train),np.max(X_test),np.max(Y_train),np.max(Y_test))
    print(np.where(np.isnan(X_test)))

    svm = sklearn.svm.SVC(kernel='rbf', probability=True)
    svm.fit(X_train, Y_train)

    # use Kernel SHAP to explain test set predictions
    explainer = shap.KernelExplainer(svm.predict_proba, X_train, link="logit")
    shap_values = explainer.shap_values(X_test, nsamples=100)

    # plot the SHAP values for the Setosa output of the first instance
    shap.summary_plot(shap_values, X_train, plot_type="bar")


