import os
# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.autograd import Variable
import argparse
import loaddata
import dataloader
import model
import scoring
import scoring_char
import transformer
import transformer_char
import numpy as np
import pickle
np.random.seed(7)

parser = argparse.ArgumentParser(description='Data')
parser.add_argument('--data', type=int, default=0, metavar='N',
                    help='data: can be 0,1,2,3,5,6,7 which specify a textdata file')
parser.add_argument('--externaldata', type=str, default='', metavar='S',
                    help='External database file. Default: Empty string')
parser.add_argument('--model', type=str, default='simplernn', metavar='S',
                    help='model type(simplernn, charcnn, bilstm). LSTM as default.')
parser.add_argument('--modelpath', type=str, default='models/simplernn_0_bestmodel.dat', metavar='S',
                    help='model file path')
parser.add_argument('--power', type=int, default=0, metavar='N',
                    help='Attack power')
parser.add_argument('--batchsize', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--scoring', type=str, default='replaceone', metavar='N',
                    help='Scoring function.')
parser.add_argument('--transformer', type=str, default='homoglyph', metavar='N',
                    help='Transformer function.')
parser.add_argument('--maxbatches', type=int, default=None, metavar='B',
                    help='maximum batches of adv samples generated')
parser.add_argument('--advsamplepath', type=str, default=None, metavar='B',
                    help='advsamplepath: If default, will generate one according to parameters')
parser.add_argument('--dictionarysize', type=int, default=20000, metavar='B',
                    help='Size of the dictionary used in RNN model')
parser.add_argument('--charlength', type=int, default=1014, metavar='N',
                    help='length: default 1014')
parser.add_argument('--wordlength', type=int, default=500, metavar='N',
                    help='word length: default 500')
args = parser.parse_args()

torch.manual_seed(8)
torch.cuda.manual_seed(8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model == "charcnn":
    args.datatype = "char"
elif args.model == "simplernn":
    args.datatype = "word"
elif args.model == "bilstm":
    args.datatype = "word"

if args.externaldata!='':
    if args.datatype == 'char':
        (data,numclass) = pickle.load(open(args.externaldata,'rb'))
        testchar = dataloader.Chardata(data, getidx = True)
        test_loader = DataLoader(testchar,batch_size=args.batchsize, num_workers=4, shuffle=False)
        alphabet = trainchar.alphabet
    elif args.datatype == 'word':
        (data,word_index,numclass) = pickle.load(open(args.externaldata,'rb'))
        testword = dataloader.Worddata(data, getidx = True)
        test_loader = DataLoader(testword,batch_size=args.batchsize, num_workers=4,shuffle=False)  
else:
    if args.datatype == "char":
        (train,test,numclass) = loaddata.loaddata(args.data)
        trainchar = dataloader.Chardata(train, getidx = True)
        testchar = dataloader.Chardata(test, getidx = True)
        train_loader = DataLoader(trainchar,batch_size=args.batchsize, num_workers=4, shuffle = True)
        test_loader = DataLoader(testchar,batch_size=args.batchsize, num_workers=4, shuffle=True)
        alphabet = trainchar.alphabet
        maxlength = args.charlength
    elif args.datatype == "word":
        (train,test,tokenizer,numclass, rawtrain, rawtest) = loaddata.loaddatawithtokenize(args.data, nb_words = args.dictionarysize, datalen = args.wordlength, withraw=True)
        word_index = tokenizer.word_index
        trainword = dataloader.Worddata(train, getidx = True, rawdata = rawtrain)
        testword = dataloader.Worddata(test, getidx = True, rawdata = rawtest)
        train_loader = DataLoader(trainword,batch_size=args.batchsize, num_workers=4, shuffle = True)
        test_loader = DataLoader(testword,batch_size=args.batchsize, num_workers=4,shuffle=True)
        maxlength =  args.wordlength
if args.model == "charcnn":
    model = model.CharCNN(classes = numclass)
elif args.model == "simplernn":
    model = model.smallRNN(classes = numclass)
elif args.model == "bilstm":
    model = model.smallRNN(classes = numclass, bidirection = True)

print(model)

state = torch.load(args.modelpath)
model = model.to(device)
try:
    model.load_state_dict(state['state_dict'])
except:
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(state['state_dict'])
    model = model.module

alltimebest = 0
bestfeature = []
def recoveradv(rawsequence, index2word, inputs, advwords):
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n '
    rear_ct = len(rawsequence)
    advsequence = rawsequence[:]
    try:
        for i in range(inputs.size()[0]-1,-1,-1):
            wordi = index2word[inputs[i].item()]
            rear_ct = rawsequence[:rear_ct].rfind(wordi)
                # print(rear_ct)
            if inputs[i].item()>=3:
                advsequence = advsequence[:rear_ct] + advwords[i] + advsequence[rear_ct + len(wordi):]
    except:
        print('something went wrong')
    return advsequence
    
def attackchar(maxbatch = None):
    corrects = .0
    total_loss = 0
    model.eval()
    tgt = []
    adv = []
    origsample = []
    origsampleidx = []
    modified = []
    for dataid, data in enumerate(test_loader):
        print(dataid)
        if maxbatch!=None and dataid >= maxbatch:
            break
        inputs,target,idx,raw = data
        inputs, target = inputs.to(device), target.to(device)
        output = model(inputs)
        tgt.append(target)
        origsample.append(inputs)
        origsampleidx.append(idx)
        pred = torch.max(output, 1)[1].view(target.size())
        losses = torch.zeros(inputs.size()[0],inputs.size()[2])
       
        losses = scoring_char.scorefunc(args.scoring)(model, inputs, pred, numclass)
        
        sorted, indices = torch.sort(losses,dim = 1,descending=True)
        advinputs = inputs.clone()
        dt = inputs.sum(dim=1).int()
        for k in range(inputs.size()[0]):
            md = raw[k][:]
            md = md[::-1]
            j=0
            t=0
            while j < args.power and t<inputs.size()[2]:
                if dt[k,indices[k][t]].item()>0:
                    advinputs[k,:,indices[k][t]],nowchar = transformer_char.transform(args.transformer)(inputs, torch.max(advinputs[k,:,indices[k][t]],0)[1].item(), alphabet)
                    md = md[:indices[k][t].item()]+nowchar+md[indices[k][t].item()+1:]
                    j+=1
                t+=1
            md = md[::-1]
            modified.append(md)
        adv.append(advinputs)        
        inputs2 = advinputs
        output2 = model(inputs2)
        pred2 = torch.max(output2, 1)[1].view(target.size())
        corrects += (pred2 == target).sum().item()
        for k in range(inputs.size()[0]):
            print(raw[k])
            print(pred[k].item())
            print(modified[k-inputs.size()[0]])
            print(pred2[k].item())
        
    target = torch.cat(tgt)
    advinputs = torch.cat(adv)
    origsamples = torch.cat(origsample)
    origsampleidx = torch.cat(origsampleidx)
    acc = corrects/advinputs.size(0)
    print('Accuracy %.5f' % (acc))
    f = open('attack_log.txt','a')
    f.write('%d\t%s\t%s\t%s\t%d\t%.2f\n' % (args.data,args.model,args.scoring,args.transformer,args.power,100*acc))
    if args.advsamplepath == None:
        advsamplepath = 'advsamples/%s_%d_%s_%s_%d.dat' % (args.model,args.data,args.scoring,args.transformer,args.power)
    else:
        advsamplepath = args.advsamplepath
    torch.save({'original':origsamples,'sampleid':origsampleidx,'advinputs':advinputs,'labels':target, 'adv_str':modified}, advsamplepath)

def attackword(maxbatch = None):
    corrects = .0
    total_loss = 0
    model.eval()
    wordinput = []
    tgt = []
    adv = []
    origsample = []
    origsampleidx = []
    
    for dataid, data in enumerate(test_loader):
        print(dataid)
        if maxbatch!=None and dataid >= maxbatch:
            break
        inputs,target, idx, raw = data
        inputs, target = inputs.to(device), target.to(device)
        origsample.append(inputs)
        origsampleidx.append(idx)
        tgt.append(target)
        wtmp = []
        output = model(inputs)
        pred = torch.max(output, 1)[1].view(target.size())
        
        losses = scoring.scorefunc(args.scoring)(model, inputs, pred, numclass)
        
        sorted, indices = torch.sort(losses,dim = 1,descending=True)

        advinputs = inputs.clone()
        
        for k in range(inputs.size()[0]):
            wtmp.append([])
            for i in range(inputs.size()[1]):
                if advinputs[k,i].item()>3:
                    wtmp[-1].append(index2word[advinputs[k,i].item()])
                else:
                    wtmp[-1].append('')
        for k in range(inputs.size()[0]):
            j = 0
            t = 0
            while j < args.power and t<inputs.size()[1]:
                if advinputs[k,indices[k][t]].item()>3:
                    word, advinputs[k,indices[k][t]] = transformer.transform(args.transformer)(advinputs[k,indices[k][t]].item(),word_index,index2word, top_words = args.dictionarysize)
                    wtmp[k][indices[k][t]] = word
                    print(word)
                    j+=1
                t+=1
        adv.append(advinputs)
        
        output2 = model(advinputs)
        pred2 = torch.max(output2, 1)[1].view(target.size())
        corrects += (pred2 == target).sum().item()
        for i in range(len(wtmp)):
            print(raw[i])
            print(pred[i].item())
            wordinputi = recoveradv(raw[i],index2word,inputs[i], wtmp[i])
            print(wordinputi)
            wordinput.append(wordinputi)
            print(pred2[i].item())
            
    target = torch.cat(tgt)
    advinputs = torch.cat(adv)
    origsamples = torch.cat(origsample)
    origsampleidx = torch.cat(origsampleidx)
    acc = corrects/advinputs.size(0)
    print('Accuracy %.5f' % (acc))
    f = open('attack_log.txt','a')
    f.write('%d\t%d\t%s\t%s\t%s\t%d\t%.2f\n' % (args.data,args.wordlength,args.model,args.scoring,args.transformer,args.power,100*acc))
    if args.advsamplepath == None:
        advsamplepath = 'advsamples/%s_%d_%s_%s_%d_%d.dat' % (args.model,args.data,args.scoring,args.transformer,args.power,args.wordlength)
    else:
        advsamplepath = args.advsamplepath
    torch.save({'original':origsamples,'sampleid':origsampleidx,'wordinput':wordinput,'advinputs':advinputs,'labels':target}, advsamplepath)

    


        
if args.datatype == "char":
    attackchar(maxbatch = args.maxbatches)
elif args.datatype == "word":
    index2word = {}
    index2word[0] = '[PADDING]'
    index2word[1] = '[START]'
    index2word[2] = '[UNKNOWN]'
    index2word[3] = ''
    if args.dictionarysize==20000:
        for i in word_index:
            if word_index[i]+3 < args.dictionarysize:
                index2word[word_index[i]+3]=i
    else:
        for i in word_index:
            if word_index[i] + 3 < args.dictionarysize:
                index2word[word_index[i]+3]=i  
    attackword(maxbatch = args.maxbatches)

