import numpy as np
import torch

def remove(inputs):
    tmp = torch.zeros(inputs.size()[1])
    return tmp
    
def flip(inputs):
    a = np.random.randint(70)
    tmp = torch.zeros(inputs.size()[1])
    if a!=69:
        tmp[a] = 1
    return tmp
    
def transform(name):
    if "remove" in name:
        return remove
    elif "flip" in name:
        return flip
    else:
        print('No transformer function found')
        sys.exit(1)