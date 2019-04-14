Adversarial Text Playground
==========================

This is Jennifer Fang's capstone research work that extends the Adversarial Playground project by Andrew Norton: https://github.com/QData/AdversarialDNN-Playground and Ji Gao and Yanjun Qi's DeepWordBug algorithm: https://github.com/QData/deepWordBug. The goal is to perform a similar function to Google's TensorFlow Playground, but for evasion attacks in adversarial machine learning on text data.  It is a web service that enables the user to visualize the creation of adversarial samples to neural networks. Some notable features include:

1. Position highlighter
2. Importance heatmap
3. Togglability + interactiveness
4. Modular structure


How to Add Code
----------------

The code structure for Adversarial Text Playground is very simple, modular, and easy to add to.

Backend code (machine learning algorithm):

Frontend code:




Installation
------------

There are git submodules in this repository; to clone all the needed files, please clone this entire repository, then cd into the Adversarial-Playground-Text-viz folder. 

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
$ cd AdversarialDNN-Playground-localDir
$ nohup python3 run.py &        # run in background even when logged out
```

Now use your favorite explorer to navigate to `localhost:9000`  or 'your_server_url:9000'
