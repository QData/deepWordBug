import torch
from torch.utils.data import DataLoader, Dataset

class Chardata(Dataset):
    def __init__(self, data, length=1014, space = False, backward = -1, alphabet = None, getidx = False):
        self.backward = backward
        self.length = length
        if alphabet!=None:
            self.alphabet = alphabet
        else:
            self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"/\\|_@#$%^&*~`+ =<>()[]{}"
        if space:
            self.alphabet = ' ' + self.alphabet
        self.dict_alphabet = {}
        for i in range(len(self.alphabet)):
            self.dict_alphabet[self.alphabet[i]] = i
        (self.inputs,self.labels) = (data.content,data.output)
        self.labels = torch.LongTensor(self.labels)
        self.getidx = getidx
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self,idx):
        x = self.encode(self.inputs[idx])
        y = self.labels[idx]
        if self.getidx==True:
            return x,y,idx
        else:
            return x,y
        return x,y
    def encode(self,x):
        inputs = torch.zeros((len(self.alphabet),self.length))
        if self.backward==1:
            for j in range(max(len(x)-self.length,0),len(x)):
                indexj = len(x)-j-1
                if indexj>=0:
                    if x[j] in self.dict_alphabet:
                        inputs[self.dict_alphabet[x[j]]][indexj] = 1.0
                else:
                    break
        elif self.backward==0:
            for j in range(max(len(x)-self.length,0),len(x)):
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
    def __init__(self, data, tokenizer = True, length=1014, space = False, backward = -1, getidx = False):
        self.backward = backward
        self.length = length
        (self.inputs,self.labels) = (data.content,data.output)
        self.labels = torch.LongTensor(self.labels)
        self.inputs = torch.from_numpy(self.inputs).long()
        self.getidx = getidx
        
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self,idx):
        x = self.inputs[idx]
        y = self.labels[idx]
        if self.getidx==True:
            return x,y,idx
        else:
            return x,y
if __name__ == '__main__':
    # Example for generating external dataset
    import pickle
    import loaddata
    (train,test,tokenizer,numclass) = loaddata.loaddatawithtokenize(0)
    test.content = test.content[:100]
    test.output = test.output[:100]
    word_index = tokenizer.word_index
    
    pickle.dump((test,word_index,numclass), open('textdata/ag_news_small_word.pickle','wb'))