import sys
import csv
csv.field_size_limit(2147483647)
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
class fulldataset:
    def __init__(self,filename,line=3):
        self.output = []
        self.content = []
        if (line==3):
            self.title = []
        elif (line==4):
            self.title = []
            self.answer = []
        self.columns=line
        self.loadcsv(filename)
    def loadcsv(self, filename, line=3):
        reader = csv.reader(open(filename, "rb"))
        count = 0
        for row in reader:
            if self.columns==2:
                self.output.append(int(row[0]))
                self.content.append(row[1])
            elif self.columns==3:
                self.output.append(int(row[0]))
                self.title.append(row[1])
                self.content.append(row[2])            
            elif self.columns==4:
                self.output.append(int(row[0]))
                self.title.append(row[1])
                self.content.append(row[2])                   
                self.answer.append(row[3])
                
class dataset:
    def __init__(self,filename,line=3):
        self.output = []
        self.content = []
        self.columns=line
        self.loadcsv(filename)
        
    def loadcsv(self, filename, line=3):
        reader = csv.reader(open(filename, "rb"))
        count = 0
        for row in reader:
            if self.columns==2:
                self.output.append(int(row[0])-1)
                self.content.append((row[1]).lower())
            elif self.columns==3:
                self.output.append(int(row[0])-1)
                self.content.append((row[1] + " " + row[2]).lower())           
            elif self.columns==4:
                self.output.append(int(row[0])-1)
                self.content.append((row[1] + " " + row[2] + " " + row[3]).lower())       
def loaddata(i = 0):
    datanames = ['ag_news','amazon_review_full','amazon_review_polarity','dbpedia','sogou_news','yahoo_answers','yelp_review_full','yelp_review_polarity','enron']
    lines = [3,3,3,3,3,4,2,2,2]
    classes= [4,5,2,14,5,10,5,2,2]
    trainadd = 'textdata/'+datanames[i]+'_csv/train.csv'
    testadd = 'textdata/'+datanames[i]+'_csv/test.csv'
    traindata = dataset(trainadd,lines[i])
    testdata = dataset(testadd,lines[i])
    return (traindata,testdata,classes[i])

def loaddatawithtokenize(i = 0, nb_words = 20000, start_char = 1, oov_char=2, index_from=3, withraw = False):
    (traindata,testdata,numclass) = loaddata(i)
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts(traindata.content + testdata.content)
    traindata.content = tokenizer.texts_to_sequences(traindata.content)
    testdata.content  = tokenizer.texts_to_sequences(testdata.content)
    
    if start_char==None:
        traindata.content = [[w + index_from for w in x] for x in traindata.content]
        testdata.content = [[w + index_from for w in x] for x in testdata.content]
    else:
        traindata.content = [[start_char]+[w + index_from for w in x] for x in traindata.content]
        testdata.content = [[start_char]+[w + index_from for w in x] for x in testdata.content]
    
    traindata.content = [[w if w < nb_words else oov_char for w in x] for x in traindata.content]
    testdata.content = [[w if w < nb_words else oov_char for w in x] for x in testdata.content]
    
    traindata.content = sequence.pad_sequences(traindata.content, maxlen=500)
    testdata.content = sequence.pad_sequences(testdata.content, maxlen=500)
    # traindata.content = torch.Tensor(traindata.content)
    # testdata.content = torch.Tensor(testdata.content)
    if withraw:
        return traindata,testdata,tokenizer,numclass,rawtrain,rawtest
    else:
        return traindata,testdata,tokenizer,numclass
    
if __name__ == "__main__":
    (traindata,testdata,numclass) = loaddata(8)
    print traindata.output