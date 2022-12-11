# Book Genre Classification based on book description
##  Model
1) first we need to analysis the data which is the most important part
   1) data is prepared in csv format which i read i by pandas
   2) columns are provided as "['index', 'title', 'genre', 'summary']"
   3) 'genre' column will be used as label
   4) 'summary' column will be our feature
   5) the distribution of 'genre' column shows the imbalance data
   6) as imbalanced , accuracy is not a good metric and we will check precision, recall ,f1score, confusion matrix
   7) descriptions also include non-alphabet characters which should be cleaned both for train and test data
   8) nltk library , word2vec
### data distribution 
    thriller        1023
    fantasy         876
    science        647
    history        600
    horror         600
    crime          500
    romance        111
    psychology     100
    sports         100
    travel         100

2) then, the next step is model architecture:
    1) as the supervised data is provided (sequence -label), this problem is considered as sequence modeling
   2)  as the it is a Many-to-1 architecture as recurrent problems
   3) as the input features are long, to avoid vanishing gradient problem, gated architectures like GRU (faster) and LSTM are preferred
   4) Attention and Transformer(Attention without RNN) ares also can be applied
   5) final layer is a Dense layer with softmax activation since the problem as multiclass classification
3) if confront with overfit, try regularization techniques
## 
### The structure of my code(tensorflow-keras) is as follows:
    Congig 
        Config.py   : as config reader from the json file
        config.json : contains the parameters to set, like learning rate, batch size, etc
    Data    
        data.csv    : our dataset 
    modules:
        callbacks.py  : list of callbacks that are called in training loop/or model fit
        datareader.py : read data as pandas dataframe, extract and preprocess features and labels, split train and validation set
        evaluation.py : metric for evaluation (customization)
        model.py      : the architecture of model
    train.py          : model training and saving
    inference.py      : for testing
    
## Docker 


## Django 

        