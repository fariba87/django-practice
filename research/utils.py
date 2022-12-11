import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
'''
In this helper function, we use a regular expression to remove
unwanted symbols, characters, and numbers, to set the reviews into a
standard format. We apply this helper function on the Summary column of
the data frame
'''
def clean_description(text):
    text=re.sub("[^a-zA-Z]"," ",str(text))
    return re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)

# explore maximum length of description and collecting vocabulary of words
tool =[]
def vocab_and_maxlen(db):
    vocab = []
    max_len = 0
    for i in range(len(db)):
        d = db['summary'][i]
        list_words = d.lower().split(' ')
        tool.append(len(list_words))
        if len(list_words)>max_len:
            max_len = len(list_words)
        for word in list_words:
            if word not in vocab:
                vocab.append(word)
    print('max len of descriptions = ',max_len) #5664
    vocab_dict = {word: token + 1 for token, word in enumerate(list(vocab))}
    token_dict = {token + 1: word for token, word in enumerate(list(vocab))}
    len(vocab)
    return vocab, max_len , vocab_dict , token_dict
def plot_distribution(db):
    plt.title('distribution of genre in book descriptions')
    sns.countplot(x=db['genre'].values)
    print(db['genre'].value_counts())
    ''' it is imbalance dataset
    thriller      1023
    fantasy        876
    science        647
    history        600
    horror         600
    crime          500
    romance        111
    psychology     100
    sports         100
    travel         100
    '''
def stratified_split(df, target, val_percent=0.2):
    '''
    Function to split a dataframe into train and validation sets, while preserving the ratio of the labels in the target variable
    Inputs:
    - df, the dataframe
    - target, the target variable
    - val_percent, the percentage of validation samples, default 0.2
    Outputs:
    - train_idxs, the indices of the training dataset
    - val_idxs, the indices of the validation dataset
    '''
    classes=list(df[target].unique())
    train_idxs, val_idxs = [], []
    for c in classes:
        idx=list(df[df[target]==c].index)
        np.random.shuffle(idx)
        val_size=int(len(idx)*val_percent)
        val_idxs+=idx[:val_size]
        train_idxs+=idx[val_size:]
    return train_idxs, val_idxs