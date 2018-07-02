import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import model

# Input: 
# model: the torch model
# input: the input at current stage 
#        Torch tensor with size (Batchsize,length)
# Output: score with size (batchsize, length)
def random(model, inputs, pred, classes):
    losses = torch.rand(inputs.size()[0],inputs.size()[2])
    return losses
    # Output a random list

def replaceone(model, inputs, pred, classes):
    losses = torch.zeros(inputs.size()[0],inputs.size()[2])
    for i in xrange(inputs.size()[2]):
        tempinputs = inputs.data.clone()
        tempinputs[:,:,i].zero_()
        tempinputs = Variable(tempinputs, volatile=True)
        tempoutput = model(tempinputs)
        losses[:,i] = F.nll_loss(tempoutput, pred, reduce=False).data
    return losses
    
def temporal(model, inputs, pred, classes):
    losses1 = torch.zeros(inputs.size()[0],inputs.size()[2])
    dloss = torch.zeros(inputs.size()[0],inputs.size()[2])
    for i in xrange(inputs.size()[2]):
        tempinputs = inputs.data.clone()
        if i!=inputs.size()[2]-1:
            tempinputs[:,:,i+1:].zero_()
        tempinputs = Variable(tempinputs, volatile=True)
        tempoutput = torch.exp(model(tempinputs))
        losses1[:,i] = tempoutput.data.gather(1,pred.view(-1,1).data) 
    dloss[:,0] = losses1[:,0] - 1.0/classes
    for i in xrange(1,inputs.size()[1]):
        dloss[:,i] = losses1[:,i] - losses1[:,i-1]
    return dloss
    
def temporaltail(model, inputs, pred, classes):
    losses1 = torch.zeros(inputs.size()[0],inputs.size()[2])
    dloss = torch.zeros(inputs.size()[0],inputs.size()[2])
    for i in xrange(inputs.size()[2]):
        tempinputs = inputs.data.clone()
        if i!=0:
            tempinputs[:,:,:i].zero_()
        tempinputs = Variable(tempinputs, volatile=True)
        tempoutput = torch.exp(model(tempinputs))
        losses1[:,i] = tempoutput.data.gather(1,pred.view(-1,1).data) 
    dloss[:,-1] = losses1[:,-1] - 1.0/classes
    for i in xrange(inputs.size()[2]-1):
        dloss[:,i] = losses1[:,i] - losses1[:,i+1]
    return dloss
    
def combined(model, inputs, pred, classes):
    temp = temporal(model, inputs, pred, classes)
    temptail = temporaltail(model, inputs, pred, classes)
    return (temp+temptail)/2
    
def grad(model, inputs, pred, classes):
    losses1 = torch.zeros(inputs.size()[0],inputs.size()[2])
    dloss = torch.zeros(inputs.size()[0],inputs.size()[2])
    model = model.module
    inputs = Variable(inputs.data,requires_grad=True)
    output = model(inputs)
    loss = F.nll_loss(output,pred)
    loss.backward()
    score = inputs.grad.norm(2,dim=1).data
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
