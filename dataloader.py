import torch
from torch.utils.data import DataLoader, Dataset
import torch.autograd as autograd
# import numpy as np
class Chardata(Dataset):
    def __init__(self, data, length=1014, space = False, backward = -1, alphabet = None):
        self.backward = backward
        self.length = length
        if alphabet!=None:
            self.alphabet = alphabet
        else:
            self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"/\\|_@#$%^&*~`+ =<>()[]{}"
        if space:
            self.alphabet = ' ' + self.alphabet
        self.dict_alphabet = {}
        for i in xrange(len(self.alphabet)):
            self.dict_alphabet[self.alphabet[i]] = i
        (self.inputs,self.labels) = (data.content,data.output)
        self.labels = torch.LongTensor(self.labels)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self,idx):
        x = self.encode(self.inputs[idx])
        y = self.labels[idx]
        return x,y
    def encode(self,x):
        inputs = torch.zeros((len(self.alphabet),self.length))
        if self.backward==1:
            for j in xrange(max(len(x)-self.length,0),len(x)):
                indexj = len(x)-j-1
                if indexj>=0:
                    if x[j] in self.dict_alphabet:
                        inputs[self.dict_alphabet[x[j]]][indexj] = 1.0
                else:
                    break
        elif self.backward==0:
            for j in xrange(max(len(x)-self.length,0),len(x)):
                indexj = j - max(len(x)-self.length,0)
                if indexj>=0:
                    if x[j] in self.dict_alphabet:
                        inputs[self.dict_alphabet[x[j]]][indexj] = 1.0
                else:
                    break
        else:
            for j, ch in enumerate(x[::-1]):
                if j>=self.length:
                    break
                if ch in self.dict_alphabet:
                # if self.alphabet.find(ch)!=-1:
                    inputs[self.dict_alphabet[ch]][j] = 1.0
        return inputs
        
class Worddata(Dataset):
    def __init__(self, data, tokenizer = True, length=1014, space = False, backward = -1):
        self.backward = backward
        self.length = length
        (self.inputs,self.labels) = (data.content,data.output)
        self.labels = torch.LongTensor(self.labels)
        self.inputs = torch.from_numpy(self.inputs).long()
        # if tokenizer:
        #     self.tokenize(nb_words)
        
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self,idx):
        x = self.inputs[idx]
        y = self.labels[idx]
        return x,y
    # def tokenize(self,nb_words = 20000):
    #     tokenizer = Tokenizer(num_words=nb_words, lower=True)
    #     tokenizer.fit_on_texts(self.inputs)
    #     self.inputs = tokenizer.texts_to_sequences(self.inputs)
        
    #     self.inputs = torch.Tensor(self.inputs)
if __name__ == '__main__':
    import loaddata
    # (train,test,numclass) = loaddata.loaddata(0)
    (train,test,tokenizer,numclass) = loaddata.loaddatawithtokenize(0)
    # trainchar = Chardata(train)
    # train_loader = DataLoader(trainchar,batch_size=64, num_workers=4, drop_last=False)
    # for i_batch, sample_batched in enumerate(train_loader):
    #     inputs,target = sample_batched
    #     print(inputs.size())
    #     # print(inputs[0])
    trainword = Worddata(train)
    train_loader = DataLoader(trainword,batch_size=64, num_workers=4, drop_last=False)
    for i_batch, sample_batched in enumerate(train_loader):
        inputs,target = sample_batched