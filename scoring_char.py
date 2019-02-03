import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import model
import numpy as np
import math

# Input: 
# model: the torch model
# input: the input at current stage 
#        Torch tensor with size (Batchsize,length)
# Output: score with size (batchsize, length)
def random(model, inputs, pred, classes):
    losses = torch.rand(inputs.size()[0],inputs.size()[2])
    return losses

def replaceone(model, inputs, pred, classes):
    losses = torch.zeros(inputs.size()[0],inputs.size()[2])
    with torch.no_grad():
        for i in range(inputs.size()[2]):
            tempinputs = inputs.clone()
            tempinputs[:,:,i].zero_()
            tempoutput = model(tempinputs)
            losses[:,i] = F.nll_loss(tempoutput, pred, reduce=False)
    return losses
    
def temporal(model, inputs, pred, classes):
    losses1 = torch.zeros(inputs.size()[0],inputs.size()[2])
    dloss = torch.zeros(inputs.size()[0],inputs.size()[2])
    for i in range(inputs.size()[2]):
        tempinputs = inputs.clone()
        if i!=inputs.size()[2]-1:
            tempinputs[:,:,i+1:].zero_()
        with torch.no_grad():
            tempoutput = torch.exp(model(tempinputs))
        losses1[:,i] = tempoutput.gather(1,pred.view(-1,1)).view(-1)
    dloss[:,0] = losses1[:,0] - 1.0/classes
    for i in range(1,inputs.size()[1]):
        dloss[:,i] = losses1[:,i] - losses1[:,i-1]
    return dloss
    
def temporaltail(model, inputs, pred, classes):
    losses1 = torch.zeros(inputs.size()[0],inputs.size()[2])
    dloss = torch.zeros(inputs.size()[0],inputs.size()[2])
    for i in range(inputs.size()[2]):
        tempinputs = inputs.clone()
        if i!=0:
            tempinputs[:,:,:i].zero_()
        with torch.no_grad():
            tempoutput = torch.exp(model(tempinputs))
        losses1[:,i] = tempoutput.gather(1,pred.view(-1,1)).view(-1)
    dloss[:,-1] = losses1[:,-1] - 1.0/classes
    for i in range(inputs.size()[2]-1):
        dloss[:,i] = losses1[:,i] - losses1[:,i+1]
    return dloss
    
def combined(model, inputs, pred, classes):
    temp = temporal(model, inputs, pred, classes)
    temptail = temporaltail(model, inputs, pred, classes)
    return (temp+temptail)/2
    
def grad(model, inputs, pred, classes):
    losses1 = torch.zeros(inputs.size()[0],inputs.size()[2])
    dloss = torch.zeros(inputs.size()[0],inputs.size()[2])
    inputs1 = inputs.clone()
    inputs1.requires_grad_(True)
    output = model(inputs1)
    loss = F.nll_loss(output,pred)
    loss.backward()
    score = inputs1.grad.norm(2,dim=1)
    return score

def scorefunc(name):
    if "temporal" in name:
        return temporal
    elif "tail" in name:
        return temporaltail
    elif "combined" in name:
        return combined
    elif "replaceone" in name:
        return replaceone
    elif "random" in name:
        return random
    elif "grad" in name:
        return grad
    else:
        print('No scoring function found')
        sys.exit(1)
