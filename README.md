# We update DeepWordBug into a new and more comprehensive CodeRepo : TextAttack for generating adversarial examples to fool NLP predictive models. 

#### TextAttack Wiki: https://textattack.readthedocs.io/en/latest/
#### TextAttack Github:  https://github.com/QData/TextAttack 
#### TextAttack Arxiv Prepint: https://arxiv.org/abs/2005.05909
TextAttack is a Python framework for running adversarial attacks against NLP models. TextAttack builds attacks from four components: a search method, goal function, transformation, and set of constraints. TextAttack's modular design makes it easily extensible to new NLP tasks, models, and attack strategies. TextAttack currently supports attacks on models trained for classification, entailment, and translation.

# DeepWordBug

This repository contains the source code of DeepWordBug, an algorithm that generates efficient adversarial samples on text input. The algorithm can attack both Char and Word model in a fast and black-box manner.


<img src="https://github.com/QData/deepWordBug/blob/master/about/example.gif" alt="example">




## Citation: 

### Title: Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers 

#### at 2018 IEEE Security and Privacy Workshops (SPW),

+ Our Presentation @ [PDF](https://github.com/QData/deepWordBug/blob/master/about/2018_Ji_DLS_presentation.pdf)

#### Extended PDF at [https://arxiv.org/abs/1801.04354](https://arxiv.org/abs/1801.04354)

+ Local version @ [PDF](https://github.com/QData/deepWordBug/blob/master/about/Ji2017_EvadeNLP-extended.pdf)


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
```bash
python attack_interactive.py --data [0-7]
```

### Prerequists:

Dataset: "Text Understanding from Scratch" dataset.

Download from [here](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M):
 
* How to use these datasets:

```bash
mkdir textdata
cd textdata
tar -xvf *.tar.gz
```

* How to set up environment

Recommand: Use anaconda. With anaconda:

```bash
conda create -n python3 python==3.6 pytorch==0.4.1
```

## Usage:

```bash
python train.py --data [0-7] --model [modelname]  ### Train the models that can be used in further attack

#--data [0-7] #select which data to use 
#--model [simplernn, bilstm, charcnn] #select the model type to train. 
# The code will automatically choose the preprocessing of the model.
``` 


```bash
python attack.py --data [0-7] --model [modelname] --modelpath [modelpath] --power [power] --scoring [algorithm] 
--transformer [algorithm] --maxbatches [batches=20] --batchsize [batchsize=128] ### Generate DeepWordBug adversarial samples
#--modelpath [modelpath] #Model path, stored by train.py
#--scoring [combined, temporal, tail, replaceone, random, grad] # Scoring algorithm
#--transformer [swap, flip, insert, remove, homoglyph] # transformer algorithm
#--power [power] # Attack power(integer, in (0,30]) which is number of modified tokens, i.e., the edit distance
#--maxbatches [batches=20] # Number of batches of adversarial samples generated, samples are selected randomly. 
# Since some test dataset is very large, to evaluate the performance we add this parameter
# to generate on parts of data. By default it will generate 2560 samples.
```


## Interactive Visualization Tool to Visualize DeepWordBug:

We build an interactive extension to visualize DeepWordbug:  
#### Code @ [https://github.com/QData/deepWordBug/tree/master/Adversarial-Playground-Text-viz](https://github.com/QData/deepWordBug/tree/master/Adversarial-Playground-Text-viz) 

## Interactive Live Demo ScreenShot

<img src="https://github.com/QData/deepWordBug/blob/master/about/demo.png" alt="demo">


## Details: 


DeepWordBug presents novel scoring strategies, outlined below, to identify critical tokens in a text sequence input that, if modified, can cause classifiers to incorrectly classify inputs with a high probability. Simple character-level transformations are applied to those highest-ranked critical tokens to minimize the differences, also known as the edit distance, between the original input and the generated adversarial example. For most effective usage, please input a text sequence of at least 5 words. 


Notable parameters when using the DeepWordBug include 
- the models, 
- the transformer algorithms, 
- the scoring algorithms, 
- and the power parameter (upper bound of number of tokens being allowed to modify).

#### Models: 

We have tried DeepWordBug on seven real-world text classification datasets. On each dataset, we have trained CharCNN and BiLSTM DNN models. 
- (0) AG News: Inputs are classified into typical news category. 
- (1) Amazon Review (Full): Full means that a full rating system between 1 and 5 stars is used. 
- (2) Amazon Review (Polarity): Polarity is more simplistic than the Full system since it only classifies inputs as Negative or Positive 
- (3) DBPedia: Inputs are classified into the encyclopedia topic category that fits the input the best. 
- (4) Yahoo Answers: Inputs are classified into the category they would be placed in if a question were asked on Yahoo Answers. 
-  (5) Yelp Review (Full): 1-5 Stars 
-  (6) Yelp Review (Polarity): Negative or positive 


+ Seven pretrained biLSTM RNN models are shared [Here](https://github.com/QData/deepWordBug/tree/master/Adversarial-Playground-Text-viz/webapp/models/models)


#### Scoring Algorithm: 

- (1) Combined: Combination of the next two options Combined_score(x) = THS(x) + λ(TTS(x)) 
- (2) Temporal: aka Temporal Head Score (THS) is the difference between the model’s prediction score as it reads up to the ith token and the model’s prediction score as it reads up to the (i-1)th token. 
- (3) Tail: aka Temporal Tail Score (TTS) is the complement of the THS. TTS computes the difference between two trailing parts of an input sequence where one contains the token and the other does not. 
- (4) Replaceone: Score of the sequence with and without the token. 
- (5) Random (baseline): This scoring function randomly selects tokens as targets. In other words, it has no method to determine which tokens to attack. 
- (6) Gradient (baseline): Contrary to random selection which uses no knowledge of the model, we also compare to full knowledge of the model, where gradients are used to find the most important tokens. 

#### Transformer Algorithm: 

- (1) Homoglyph: Replace character with a similar-looking, yet different character. This is the best performing transformer. 

> This is a newly added transformer based on the homoglyph attack. A homoglyph attack is an attacking method that uses symbols with identical shapes. The following figure shows a table including all the text characters together with its homoglyph pair in our design. 

<img src="https://github.com/QData/deepWordBug/blob/master/about/homoglyph.png" width="242" alt="demo">


- (2) Swap: Swap two adjacent letters in the word. 
- (3) Substitution: Substitute a letter in the word with a random letter. 
- (4) Deletion: Delete a random letter from the word. The deletion is displayed with a red _ underscore character. 
- (5) Insertion: Insert a random letter in the word. 


#### Power: 

The number of tokens per text sequence to be modified. If this number is higher than the length of the text sequence, then all the words in the sequence will be modified.


