# -*- coding: utf-8 -*-
import numpy as np
import torch
import sys

def remove(inputs, char, alphabet):
    tmp = torch.zeros(inputs.size()[1])
    nowchar = alphabet[0]
    return tmp,nowchar
    
def flip(inputs, char, alphabet):
    a = np.random.randint(70)
    nowchar = alphabet[a]
    tmp = torch.zeros(inputs.size()[1])
    if a!=69:
        tmp[a] = 1
    return tmp, nowchar

homos = {' ':u'\u00A0','-':'˗','9':'৭','8':'Ȣ','7':'𝟕','6':'б','5':'Ƽ','4':'Ꮞ','3':'Ʒ','2':'ᒿ','1':'l','0':'O',"'":'`','a': 'ɑ', 'b': 'Ь', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏', 'g': 'ɡ', 'h': 'հ', 'i': 'і', 'j': 'ϳ', 'k': '𝒌', 'l': 'ⅼ', 'm': 'ｍ', 'n': 'ո', 'p': 'р', 'q': 'ԛ', 'r': 'ⲅ', 's': 'ѕ', 't': '𝚝', 'u': 'ս', 'v': 'ѵ', 'w': 'ԝ', 'x': '×', 'y': 'у', 'z': 'ᴢ'}
def homoglyph(inputs, char, alphabet):
    if alphabet[char] in homos:
        nowchar = homos[alphabet[char]]
    else:
        nowchar = alphabet[char]
    tmp = torch.zeros(inputs.size()[1])
    if nowchar in alphabet:
        tmp[alphabet.index(nowchar)] = 1
    return tmp,nowchar
    
def transform(name):
    if "remove" in name:
        return remove
    elif "flip" in name:
        return flip
    elif "homoglyph" in name:
        return homoglyph
    else:
        print('No transformer function found')
        sys.exit(1)