Adversarial Text Playground
==========================

This is Jennifer Fang's capstone research work that extends the Adversarial Playground project by Andrew Norton: https://github.com/QData/AdversarialDNN-Playground and Ji Gao and Yanjun Qi's DeepWordBug algorithm: https://github.com/QData/deepWordBug. 

The goal is to perform a similar function to Google's TensorFlow Playground, but for visualizations of evasion attacks in adversarial machine learning on text data as well as other machine learning classifiers.  This is a web service that enables the user to visualize the creation of adversarial samples to neural networks. Some notable features include:

1. Position highlighter
2. Importance heatmap
3. Togglability + interactiveness
4. Modular structure


How to Add Code
----------------

The code structure for Adversarial Text Playground is very simple, modular, and easy to add to.

<strong>Backend code </strong>(machine learning algorithm): place all relevant files in the webapp/models folder. Connect the model to the frontend code in the views.py file. Follow the example given for the deepwordbug model. 

<strong>Frontend code </strong>(for the actual visualization): place this in the webapp/templates folder. One template file per algorithm. Each new template file represents a new tab on the top navigation bar.  

The index.html file can be easily added to for any additional algorithms. The base.html file provides a guideline for how to achieve all the features listed above. The deepwordbug.html file demonstrates how to add a tab for a new algorithm in Adversarial-Text-Playground. There are extensive comments throughout each file to show exactly how and where code should be added. 



Installation
------------

If you choose to install this locally and not use the AWS EB link above, there are git submodules in this repository; to clone all the needed files, please clone this entire repository, then cd into the Adversarial-Playground-Text-viz folder. 

The primary requirements for this package are Python 3 with Tensorflow version 1.0.1 or greater and != version 1.7.  The `requirements.txt` file contains a listing of the required Python packages; to install all requirements, run the following:

```
pip3 -r install requirements.txt
```

If the above command does not work, use the following:

```
pip3 install -r requirements.txt
```

Or use the following instead if need to sudo:
```
sudo -H pip  install -r requirements.txt
```

Use:
----

### To Deploy the webserver:

Once you've downloaded the repo, cd into Adversarial-Playground-Text-viz, then run `python3 run.py` :

```
$ cd Adversarial-Playground-Text-viz
$ python3 run.py &       
```

Or run the following command to run the webapp in the background even when logged out from the server:
```
$ cd Adversarial-Playground-Text-viz
$ nohup python3 run.py &        # run in background even when logged out
```

Now use your favorite explorer to navigate to `localhost:9000`  or 'your_server_url:9000'


## This Interactive Visualization Tool to Visualize DeepWordBug:

This folder builds an interactive extension to visualize DeepWordbug:  
#### Code @ [https://github.com/QData/deepWordBug/tree/master/Adversarial-Playground-Text-viz](https://github.com/QData/deepWordBug/tree/master/Adversarial-Playground-Text-viz) 

## Interactive Live Demo @  

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
