import numpy as np
np.random.seed(7)

def swap(wordid,word_index,index2word,top_words=20000):
    word = index2word[wordid]
    if len(word)!=1:
        s = np.random.randint(0,len(word)-1)
        cword = word[:s] + word[s+1] + word[s] + word[s+2:]
        if cword in word_index:
            wid = word_index[cword] + 3
            if wid>=top_words:
                wid = 2
        else:
            wid = 2
    else:
        cword = word
        # cword = word + '0'
        if cword in word_index:
            wid = word_index[cword] + 3
            if wid>=top_words:
                wid = 2
        else:
            wid = 2
    return (cword,wid)
    
def flip(wordid,word_index,index2word,top_words=20000):
    word = index2word[wordid]
    s = np.random.randint(0,len(word))
    letter = ord(word[s])
    rletter = np.random.randint(0,25)+97
    if rletter >= letter:
        rletter += 1
    cword = word[:s] + chr(rletter) + word[s+1:]
    if cword in word_index:
        wid = word_index[cword] + 3
        if wid >= top_words:
            wid = 2
    else:
        wid = 2
    return (cword,wid)

def f2(wordid,word_index,index2word,top_words=20000):
    word = index2word[wordid]
    s = np.random.randint(0,len(word))
    letter = ord(word[s])
    rletter = np.random.randint(0,25)+97
    if rletter >= letter:
        rletter += 1
    cword = word[:s] + chr(rletter) + word[s+1:]
    if len(word)>1:
        s2 = np.random.randint(0,len(word)-1)
        if s2>=s:
            s2+=1
        letter = ord(word[s2])
        rletter = np.random.randint(0,25)+97
        if rletter >= letter:
            rletter += 1
        cword = cword[:s2] + chr(rletter) + cword[s2+1:]
    if cword in word_index:
        wid = word_index[cword] + 3
        if wid>=top_words:
            wid = 2
    else:
        wid = 2
    return (cword,wid)
    
def remove(wordid,word_index,index2word,top_words=20000):
    word = index2word[wordid]
    s = np.random.randint(0,len(word))
    if len(word)>1:
        cword = word[:s] + word[s+1:]
    else:
        cword = word
    if cword in word_index:
        wid = word_index[cword] + 3
        if wid>=top_words:
            wid = 2
    else:
        wid = 2
    return (cword,wid)

def remove2(wordid,word_index,index2word,top_words=20000):
    word = index2word[wordid]
    s = np.random.randint(0,len(word))
    if len(word)>1:
        cword = word[:s] + word[s+1:]
    else:
        cword = word
    if len(cword)>1:
        s = np.random.randint(0,len(cword))
        cword = cword[:s] + cword[s+1:]
    if cword in word_index:
        wid = word_index[cword] + 3
        if wid>=top_words:
            wid = 2
    else:
        wid = 2
    return (cword,wid)
    
def insert(wordid,word_index,index2word,top_words=20000):
    word = index2word[wordid]
    s = np.random.randint(0,len(word)+1)
    cword = word[:s] + chr(97+np.random.randint(0,26)) + word[s:]
    if cword in word_index:
        wid = word_index[cword] + 3
        if wid>=top_words:
            wid = 2
    else:
        wid = 2
    return (cword,wid)

def transform(name):
    if "swap" in name:
        return swap
    elif "flip" in name:
        return flip
    elif "f2" in name:
        return f2
    elif "insert" in name:
        return insert
    elif "remove" in name:
        return remove
    elif "r2" in name:
        return remove2
    else:
        print('No transformer function found')
        sys.exit(1)