# Citation: 

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

* This is an implementation of our deepWordBug attack on text data. The code is written in python 2.7. It requires pytorch 0.3 and keras utils to run our code.

### Prerequists:

Dataset: "Text Understanding from Scratch" dataset.

Download from [here](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M):
 
* How to use these datasets:

`mkdir textdata`

`cd textdata`

`tar -xvf *.tar.gz` 


## Usage:
`train.py --data [0-7] --model [modelname]` ### Train the models that can be used in further attack

`--data [0-7]` #select which data 

`--model [simplernn, bilstm, charcnn]` #select the model type to train. The code will automatically choose the preprocessing of the model.

`attack.py --data [0-7] --model [modelname] --modelpath [modelpath] --power [power] --scoring [algorithm] --transformer [algorithm]` ### Generate DeepWordBug adversarial samples

`-- modelpath [modelpath]` #Model path, stored by train.py

`-- scoring [combined, temporal, tail, replaceone, random, grad]` # Scoring algorithm

`-- transformer [swap, flip, insert, remove]` # transformer algorithm

`-- power [power]` # Attack power(integer, in (0,30]) which is number of modified tokens, i.e., the edit distance

