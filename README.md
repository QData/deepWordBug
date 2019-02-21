# DeepWordBug

This repository contains the source code of DeepWordBug, an algorithm that generates efficient adversarial samples on text input. The algorithm can attack both Char and Word model in a fast and black-box manner.

<img src="https://github.com/QData/deepWordBug/blob/master/example.gif" alt="example">

## Citation: 

### Title: Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers 

#### at 2018 IEEE Security and Privacy Workshops (SPW),

#### PDF at [https://arxiv.org/abs/1801.04354](https://arxiv.org/abs/1801.04354)

```
@INPROCEEDINGS{JiDeepWordBug18, 
author={J. Gao and J. Lanchantin and M. L. Soffa and Y. Qi}, 
booktitle={2018 IEEE Security and Privacy Workshops (SPW)}, 
title={Black-Box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers}, 
year={2018}, 
pages={50-56}, 
keywords={learning (artificial intelligence);pattern classification;program debugging;text analysis;deep learning classifiers;character-level transformations;IMDB movie reviews;Enron spam emails;real-world text datasets;scoring strategies;text input;text perturbations;DeepWordBug;black-box attack;adversarial text sequences;black-box generation;Perturbation methods;Machine learning;Task analysis;Recurrent neural networks;Prediction algorithms;Sentiment analysis;adversarial samples;black box attack;text classification;misclassification;word embedding;deep learning}, 
doi={10.1109/SPW.2018.00016}, 
ISSN={}, 
month={May},}
```

## Dependency

### The code is written in python 3. It requires [Pytorch](pytorch.org) 0.4 or higher to run our code.

### Quick start

#### Use attack_interactive.py to get some ideas of our attack!
Usage:
```
attack_interactive.py --data [0-7]
```

### Prerequists:

Dataset: "Text Understanding from Scratch" dataset.

Download from [here](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M):
 
* How to use these datasets:

```
mkdir textdata
cd textdata
tar -xvf *.tar.gz
```

* How to set up environment

Recommand: Use anaconda. With anaconda:

```
conda create -n python3 python==3.6 pytorch==0.4.1
```

## Usage:

```
train.py --data [0-7] --model [modelname]  ### Train the models that can be used in further attack

--data [0-7] #select which data to use 
--model [simplernn, bilstm, charcnn] #select the model type to train. The code will automatically choose the preprocessing of the model.
``` 


```
attack.py --data [0-7] --model [modelname] --modelpath [modelpath] --power [power] --scoring [algorithm] --transformer [algorithm] --maxbatches [batches=20] --batchsize [batchsize=128] ### Generate DeepWordBug adversarial samples
--modelpath [modelpath] #Model path, stored by train.py
--scoring [combined, temporal, tail, replaceone, random, grad] # Scoring algorithm
--transformer [swap, flip, insert, remove] # transformer algorithm
--power [power] # Attack power(integer, in (0,30]) which is number of modified tokens, i.e., the edit distance
--maxbatches [batches=20] # Number of batches of adversarial samples generated, samples are selected randomly. Since some test dataset is very large, to evaluate the performance we add this parameter to generate on parts of data. By default it will generate 2560 samples.
```
