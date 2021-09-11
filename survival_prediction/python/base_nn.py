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
    'batch_size': 8,
    'lr': 0.001,
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
            if i > 18:
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
            if i > 18:
                _input = np.concatenate((_input, _input_arr[i]), axis=None)
        _input = torch.from_numpy(np.array(_input)).float()
        _target = torch.from_numpy(np.array(_gt)).long()

        return _input, _target

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    train_file = 'Train_dir.txt'
    test_file = 'Val_dir.txt'
    train_dataset = HEMDataset(text_dir=train_file)
    test_dataset = HEMDataset_test(text_dir=test_file)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2,drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2,drop_last=False)

    net = Net().cuda()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args['lr'])
    max_epoch = 50
    for epoch in range (max_epoch):
        net.train()
        for batch_idx, data in enumerate(train_loader):
            inputs, labels = data
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        net.eval()
        correct, total = 0, 0
        class_pred, class_gt = [], []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = net(inputs)

                _, predicted = torch.max(outputs.data, 1)
                class_pred.append(predicted.item())
                class_gt.append(targets.item())
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print('Epoch:', epoch)#, 'Accuracy: %f %%' % (100 * correct / total))
        print(confusion_matrix(np.array(class_pred),np.array(class_gt)))
        print(classification_report(np.array(class_pred),np.array(class_gt)))
        print(accuracy_score(np.array(class_pred),np.array(class_gt)))
        print('')
    print('Finished Training')
