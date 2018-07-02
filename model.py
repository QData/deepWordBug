import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CharCNN(nn.Module):
    def __init__(self, classes=4, num_features=69, dropout=0.5):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_features, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )            
            
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()    
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
            
        
        self.fc1 = nn.Sequential(
            nn.Linear(8704, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.fc3 = nn.Linear(1024, classes)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.log_softmax(x)
        
        return x
        
class smallRNN(nn.Module):
    def __init__(self, classes=4, bidirection = False, layernum=1, length=20000,embedding_size =100, hiddensize = 100):
        super(smallRNN, self).__init__()
        self.embd = nn.Embedding(length, embedding_size)
        # self.lstm = nn.LSTMCell(hiddensize, hiddensize)
        self.lstm = nn.LSTM(embedding_size, hiddensize, layernum, bidirectional = bidirection)
        self.hiddensize = hiddensize
        numdirections = 1 + bidirection
        self.hsize = numdirections * layernum
        self.linear = nn.Linear(hiddensize * numdirections, classes)
        self.log_softmax = nn.LogSoftmax()
    def forward(self, x, returnembd = False):
        embd = self.embd(x)
        if returnembd:
            embd = Variable(embd.data, requires_grad=True).cuda()
            embd.retain_grad()
            # print embd.size()
        h0 = Variable(torch.zeros(self.hsize, embd.size(0), self.hiddensize)).cuda()
        c0 = Variable(torch.zeros(self.hsize, embd.size(0), self.hiddensize)).cuda()
        # for inputs in x:
        x = embd.transpose(0,1)
        x,(hn,cn) = self.lstm(x,(h0,c0))
        x = x[-1]
        # x = x[-1].transpose(0,1)
        # x = x.view(x.size(0),-1)
        x = self.log_softmax(self.linear(x))
        if returnembd:
            return embd,x
        else:
            return x
            
class smallcharRNN(nn.Module):
    def __init__(self, classes=4, bidirection = False, layernum=1, char_size = 69, hiddensize = 100):
        super(smallcharRNN, self).__init__()
        # self.embd = nn.Embedding(length, embedding_size)
        # self.lstm = nn.LSTMCell(hiddensize, hiddensize)
        self.lstm = nn.LSTM(char_size, hiddensize, layernum, bidirectional = bidirection)
        self.hiddensize = hiddensize
        numdirections = 1 + bidirection
        self.hsize = numdirections * layernum
        self.linear = nn.Linear(hiddensize * numdirections, classes)
        self.log_softmax = nn.LogSoftmax()
    def forward(self, x):
        h0 = Variable(torch.zeros(self.hsize, x.size(0), self.hiddensize)).cuda()
        c0 = Variable(torch.zeros(self.hsize, x.size(0), self.hiddensize)).cuda()
        # for inputs in x:
        x = x.transpose(0,1)
        x = x.transpose(0,2)
        x,(hn,cn) = self.lstm(x,(h0,c0))
        x = x[-1]
        x = self.log_softmax(self.linear(x))
        # x = x[-1].transpose(0,1)
        # x = x.view(x.size(0),-1)
        return x
        
class WordCNN(nn.Module):
    def __init__(self, classes=4, num_features=100, dropout=0.5, maxword = 20000):
        super(WordCNN, self).__init__()
        self.embd = nn.Embedding(maxword, num_features)
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_features, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )            
            
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()    
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
            
        
        self.fc1 = nn.Sequential(
            nn.Linear(3584, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.fc3 = nn.Linear(1024, classes)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.embd(x)
        x = x.transpose(1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        # print x.size()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.log_softmax(x)
        
        return x