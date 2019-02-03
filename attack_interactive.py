import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import model
import scoring
import scoring_char
import transformer
import transformer_char
import numpy as np
import pickle

default_filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

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
    
def simple_tokenize(input_seq, dict_word, filters= default_filter):
    input_seq = input_seq.lower()
    translate_dict = dict((c, ' ') for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = input_seq.translate(translate_map)
    seq = text.strip().split(' ')
    index_seq = []
    for i in seq:
        if i in dict_word:
            if dict_word[i]+3<20000:
                index_seq.append(dict_word[i]+3)
            else:
                index_seq.append(2)
        else:
            index_seq.append(2)
    return index_seq

default_alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"/\\|_@#$%^&*~`+ =<>()[]{}"
def transchar(x, alphabet=default_alphabet,length=1014):
    inputs = torch.zeros(1,len(alphabet),length)
    for j, ch in enumerate(x[::-1]):
        if j>=length:
            break
        if ch in alphabet:
            inputs[0,alphabet.find(ch),j] = 1.0
    return inputs
    
def visualize(input_str, dict_word=[],index2word = [], classes_list = [], power=5,  scoring_alg = 'replaceone', transformer_alg = 'homoglyph', model = model, mode = 'word', maxlength = 500, device = None, filter_char = default_filter, alphabet = default_alphabet):
    numclass = len(classes_list)

    if mode=='word':

        input_seq = simple_tokenize(input_str, dict_word)
        input_seq = torch.Tensor(input_seq).long().view(1,-1)
        if device:
            input_seq = input_seq.to(device)
        res1 = model(input_seq)
        pred1 = torch.max(res1, 1)[1].view(-1)
        losses = scoring.scorefunc(scoring_alg)(model, input_seq, pred1, numclass)
        
        print(input_str)
        pred1 = pred1.item()
        print('original:',classes_list[pred1])
        
        sorted, indices = torch.sort(losses,dim = 1,descending=True)

        advinputs = input_seq.clone()
        wtmp = []
        for i in range(input_seq.size()[1]):
            if advinputs[0,i].item()>3:
                wtmp.append(index2word[advinputs[0,i].item()])
            else:
                wtmp.append('')
        j = 0
        t = 0
        while j < power and t<input_seq.size()[1]:
            if advinputs[0,indices[0][t]].item()>3:
                word, advinputs[0,indices[0][t]] = transformer.transform(transformer_alg)(advinputs[0,indices[0][t]].item(),dict_word, index2word, top_words = 20000)
                wtmp[indices[0][t]] = word
                j+=1
            t+=1
        output2 = model(advinputs)
        pred2 = torch.max(output2, 1)[1].view(-1).item()
        adv_str = recoveradv(input_str.lower(), index2word, input_seq[0], wtmp)
        print(adv_str)
        print('adversarial:', classes_list[pred2])
        return (input_str, torch.exp(res1).detach().cpu()[0], classes_list[pred1], adv_str, torch.exp(output2).detach().cpu()[0], classes_list[pred2])
    elif mode=='char':
        inputs = transchar(input_str, alphabet = alphabet)
        if device:
            inputs = inputs.to(device)
        output = model(inputs)
        pred1 = torch.max(output, 1)[1].view(-1)
        
        losses = scoring_char.scorefunc(scoring_alg)(model, inputs, pred1, numclass)
        
        sorted, indices = torch.sort(losses,dim = 1,descending=True)
        advinputs = inputs.clone()
        dt = inputs.sum(dim=1).int()
        j=0
        t=0
        md = input_str.lower()[:][::-1]
        while j < power and t<inputs.size()[2]:
            if dt[0,indices[0][t]].item()>0:
                advinputs[0,:,indices[0][t]],nowchar = transformer_char.transform(transformer_alg)(inputs, torch.max(advinputs[0,:,indices[0][t]],0)[1].item(), alphabet)
                md = md[:indices[0][t].item()] + nowchar + md[indices[0][t].item()+1:]
                j+=1
            t+=1
        md = md[::-1]
        output2 = model(advinputs)
        pred2 = torch.max(output2, 1)[1].view(-1)
        print(input_str)
        print('original:',classes_list[pred1.item()])
        print(md)
        print('adversarial:', classes_list[pred2.item()])
        return (input_str, torch.exp(output)[0].detach().cpu(), classes_list[pred1.item()], md, torch.exp(output2)[0].detach().cpu(), classes_list[pred2.item()])
    else:
        raise Exception('Wrong mode %s' % mode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data')
    parser.add_argument('--data', type=int, default=1, metavar='N',
                        help='data 0 - 7: Default: Amazon Review Full, classify the attitude 1-5 of the review')
    parser.add_argument('--model', type=str, default='simplernn', metavar='N',
                        help='model type: LSTM as default')
    parser.add_argument('--modelpath', type=str, default='', metavar='N',
                        help='model file path')
    args = parser.parse_args()
    if not args.modelpath:
        args.modelpath = 'models/%s_%d_bestmodel.dat' % (args.model,args.data)
    torch.manual_seed(8)
    torch.cuda.manual_seed(8)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model == "charcnn":
        args.datatype = "char"
    elif args.model == "simplernn":
        args.datatype = "word"
    info = pickle.load(open('dict/'+str(args.data)+'.info','rb'))
    word_index = info['word_index']
    index2word = info['index2word']
    classes_list = info['classes_list']
    numclass = len(classes_list)
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
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state['state_dict'])
        model = model.module
    print('Type input:')
    s = input()
    
    while s:
        visualize(s, power=2, mode=args.datatype, model = model, dict_word = word_index, index2word = index2word, classes_list = classes_list, device = device)
        print('Type next input(Enter to exit):')
        s = input()