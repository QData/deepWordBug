import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import model
from . import scoring
from . import scoring_char
from . import transformer
from . import transformer_char
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
    
def visualize(input_str, model_num, power, scoring_alg, transformer_alg, dict_word=[], index2word = [], classes_list = [], 
	model_type = 'simplernn', mode = 'word', maxlength = 500, device = None, filter_char = default_filter, alphabet = default_alphabet):

    info = pickle.load(open(os.getcwd() + '/webapp/models/dict/' + model_num + '.info', 'rb'))
    dict_word = info['word_index']
    index2word = info['index2word']
    classes_list = info['classes_list']
    numclass = len(classes_list)

    viz_model = model.smallRNN(classes = numclass)

    torch.manual_seed(8)
    torch.cuda.manual_seed(8)
    modelpath = 'webapp/models/models/%s_%s_bestmodel.dat' % (model_type, model_num)

    if torch.cuda.is_available():
    	device = torch.device("cuda")
    	state = torch.load(modelpath)
    	viz_model = viz_model.to(device)
    else:
    	device = torch.device("cpu")
    	state = torch.load(modelpath, map_location = 'cpu')
    	viz_model = viz_model.to(device)
    
    try:
        model.load_state_dict(state['state_dict'])
    except:
        viz_model = torch.nn.DataParallel(viz_model)
        viz_model.load_state_dict(state['state_dict'])
        viz_model = viz_model.module

    if mode == 'word':
        input_seq = simple_tokenize(input_str, dict_word)
        input_seq = torch.Tensor(input_seq).long().view(1,-1)
        if device:
            input_seq = input_seq.to(device)
        res1 = viz_model(input_seq)
        pred1 = torch.max(res1, 1)[1].view(-1)
        losses = scoring.scorefunc(scoring_alg)(viz_model, input_seq, pred1, numclass)

        # Use the losses to get the scores and pick top 2 to display heatmap
        max_scores = losses.numpy().argsort()[0][::-1][0:2]
        print(losses)
        print(max_scores)

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
        output2 = viz_model(advinputs)
        pred2 = torch.max(output2, 1)[1].view(-1).item()
        adv_str = recoveradv(input_str.lower(), index2word, input_seq[0], wtmp)

        # 1) exp = the probability classes
        # 2) res1 = original, output2 = adversarial
        # Return: orig class, adv class, adv example, orig_likelihood, adv_likelihood, 
        #    class list (for x axis of plot), max_scores for heatmap
        return (classes_list[pred1], classes_list[pred2], adv_str, 
            torch.exp(res1).detach().cpu()[0], torch.exp(output2).detach().cpu()[0], classes_list, max_scores)

    else:
        raise Exception('Wrong mode %s' % mode)
