import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import loaddata
import dataloader
import os
import shutil
import model
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

def save_checkpoint(state, is_best, filename='checkpoint.dat'):
    torch.save(state, filename + '_checkpoint.dat')
    if is_best:
        shutil.copyfile(filename + '_checkpoint.dat', filename + "_bestmodel.dat")
        
parser = argparse.ArgumentParser(description='Train a model from text data')
parser.add_argument('--data', type=int, default=0, metavar='N',
                    help='data 0 - 8')
parser.add_argument('--seed', type=int, default=7, metavar='N',
                    help='random seed')
parser.add_argument('--length', type=int, default=1014, metavar='N',
                    help='length in char data: default 1014')
parser.add_argument('--model', type=str, default='simplernn', metavar='N',
                    help='model type: LSTM as default')
parser.add_argument('--space', type=bool, default=False, metavar='B',
                    help='Whether including space in the alphabet')
parser.add_argument('--backward', type=int, default=-1, metavar='B',
                    help='Backward direction of char data')
parser.add_argument('--epochs', type=int, default=10, metavar='B',
                    help='Number of epochs')
parser.add_argument('--batchsize', type=int, default=128, metavar='B',
                    help='batch size')
parser.add_argument('--dictionarysize', type=int, default=20000, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, metavar='B',
                    help='learning rate')
parser.add_argument('--validratio', type=float, default=0.2, metavar='B',
                    help='valid ratio')
parser.add_argument('--maxnorm', type=float, default=400, metavar='B',
                    help='learning rate')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.model == "charcnn":
    args.datatype = "char"
elif args.model == "simplernn":
    args.datatype = "word"
elif args.model == "bilstm":
    args.datatype = "word"
elif args.model == "smallcharrnn":
    args.datatype = "char"
    args.length = 300
elif args.model == "wordcnn":
    args.datatype = "word"
    
print("Loading data..")
if args.datatype == "char":
    (train,test,numclass) = loaddata.loaddata(args.data)
    traintext = dataloader.Chardata(train,backward = args.backward, length = args.length)
    testtext = dataloader.Chardata(test,backward = args.backward,length = args.length)
elif args.datatype == "word":
    (train,test,tokenizer,numclass) = loaddata.loaddatawithtokenize(args.data,nb_words = args.dictionarysize)
    traintext = dataloader.Worddata(train)
    testtext = dataloader.Worddata(test)

num_train = len(traintext)
indices = range(num_train)
split = int(np.floor(args.validratio * num_train))
np.random.seed(args.seed)
np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(traintext,batch_size=args.batchsize, num_workers=4, sampler = train_sampler)
valid_loader = DataLoader(traintext,batch_size=args.batchsize, num_workers=4, sampler = valid_sampler)
test_loader = DataLoader(testtext,batch_size=args.batchsize, num_workers=4)

if args.model == "charcnn":
    model = model.CharCNN(classes = numclass)
elif args.model == "simplernn":
    model = model.smallRNN(classes = numclass)
elif args.model == "bilstm":
    model = model.smallRNN(classes = numclass, bidirection = True)
elif args.model == "smallcharrnn":
    model = model.smallcharRNN(classes = numclass)
elif args.model == "wordcnn":
    model = model.WordCNN(classes = numclass)
    
model = model.cuda()
model = torch.nn.DataParallel(model).cuda()
print "Model:\n",model
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if not os.path.exists('models'):
    os.makedirs('models')
bestacc = 0
for epoch in xrange(args.epochs+1):
    print 'Start epoch %d' % epoch
    model.train()
    for dataid, data in enumerate(train_loader):
        inputs,target = data
        inputs,target = Variable(inputs),  Variable(target)
        inputs, target = inputs.cuda(), target.cuda()
        output = model(inputs)
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
    
    correct = .0
    total_loss = 0
    model.eval()
    for dataid, data in enumerate(valid_loader):
        inputs,target = data
        inputs,target = Variable(inputs, volatile=True),  Variable(target)
        inputs, target = inputs.cuda(), target.cuda()
        output = model(inputs)
        loss = F.nll_loss(output, target)
        total_loss += loss.data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = correct/len(valid_idx)
    avg_loss = total_loss/len(valid_idx)
    print('Epoch %d : Loss %.4f Accuracy %.5f' % (epoch,avg_loss,acc))
    is_best = acc > bestacc
    if is_best:
        bestacc = acc
    if args.dictionarysize!=20000:
        fname = "models/" + args.model +str(args.dictionarysize) + "_" + str(args.data)
    else:
        fname = "models/" + args.model + "_" + str(args.data)
        
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'bestacc': bestacc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename = fname)
    
