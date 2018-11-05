# Citation: 

### Title: Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers

### at 2018 IEEE Security and Privacy Workshops (SPW),

### PDF at [https://arxiv.org/abs/1801.04354](https://arxiv.org/abs/1801.04354)

```
@article{DBLP:journals/corr/abs-1801-04354,
  author    = {Ji Gao and
               Jack Lanchantin and
               Mary Lou Soffa and
               Yanjun Qi},
  title     = {Black-box Generation of Adversarial Text Sequences to Evade Deep Learning
               Classifiers},
  journal   = {CoRR},
  volume    = {abs/1801.04354},
  year      = {2018},
  url       = {http://arxiv.org/abs/1801.04354},
  archivePrefix = {arXiv},
  eprint    = {1801.04354},
  timestamp = {Thu, 01 Feb 2018 19:52:26 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1801-04354},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

# DeepWordBug implementation

* This is an implementation of our deepWordBug attack on text data. The code is written in python 3. It requires Pytorch 0.4  to run our code.

### Update: our code is now upgraded to Python 3 & pytorch 0.4.1

### Prerequists:

Dataset: "Text Understanding from Scratch" dataset.

Download from [here](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M):
 
* How to use these datasets:

`mkdir textdata`

`cd textdata`

`tar -xvf *.tar.gz` 

* How to set up environment

Recommand: Use anaconda. With anaconda:

`conda create -n python3 python==3.7`
`conda install pytorch==0.4.1`

## Usage:
`train.py --data [0-7] --model [modelname]` ### Train the models that can be used in further attack

`--data [0-7]` #select which data 

`--model [simplernn, bilstm, charcnn]` #select the model type to train. The code will automatically choose the preprocessing of the model.

`attack.py --data [0-7] --model [modelname] --modelpath [modelpath] --power [power] --scoring [algorithm] --transformer [algorithm] --maxbatches [batches=20] --batchsize [batchsize=128]` ### Generate DeepWordBug adversarial samples

`-- modelpath [modelpath]` #Model path, stored by train.py

`-- scoring [combined, temporal, tail, replaceone, random, grad]` # Scoring algorithm

`-- transformer [swap, flip, insert, remove]` # transformer algorithm

`-- power [power]` # Attack power(integer, in (0,30]) which is number of modified tokens, i.e., the edit distance

`--maxbatches [batches=20]` # Number of batches of adversarial samples generated, samples are selected randomly. Since some test dataset is very large, to evaluate the performance we add this parameter to generate on parts of data. By default it will generate 2560 samples.
