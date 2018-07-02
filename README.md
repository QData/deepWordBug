# DeepWordBug implementation

* This is an implementation of our deepWordBug attack on text data. It requires pytorch 0.2 and keras utils to run our code.

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

`attack.py --data [0-7] --model [modelname] --modelpath [modelpath] --scoring [algorithm] --transformer [algorithm]` ### Generate DeepWordBug adversarial samples

`-- modelpath [modelpath]` #Model path, stored by train.py

`-- scoring [combined, temporal, tail, replaceone, random, grad]` # Scoring algorithm

`-- transformer [swap, flip, insert, remove]` # transformer algorithm

