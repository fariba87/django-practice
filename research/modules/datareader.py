import pandas as pd
import numpy as np
import tensorflow as tf
from utils import clean_description, stratified_split
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
############################################################
def DataLoader(data_path, shuffle ):
    #data_dir = '../data/data.csv'
    db = pd.read_csv(data_dir)
    # db.genre
    #print(db.describe())
    unique_labels = db['genre'].unique()
    unique_labels = db[
        'genre'].unique()  # 10  #10 classes # ['fantasy' 'science' 'crime' 'history' 'horror' 'thriller' 'psychology' , 'romance' 'sports' 'travel']
    list_label = unique_labels.tolist()
    dict = {k: i for i, k in enumerate(list_label)}
    print(list(dict.keys()))
    print(dict)

    le = preprocessing.LabelEncoder()
    le.fit(unique_labels)
    list(le.classes_)
    le.transform(unique_labels)
    label_encode = []
    for i in range(len(db)):
        label_id = le.transform([db['genre'][i]])
        label_encode.append(label_id.tolist())
    Y = label_encode
    YY = tf.keras.utils.to_categorical(Y, num_classes=10, dtype='float32')

    X = db['summary'].apply(clean_description)
    '''
    tokenizer does the heavy lifting in the background, by taking care of
    duplicate/similar words, removing punctuation, and converting letters
    to lower case
    '''
    tokenizer = Tokenizer(num_words=10000, oov_token='xxxxxxx')
    tokenizer.fit_on_texts(X)
    X_dict = tokenizer.word_index
    # vocabulary
    # word2idx = tokenizer.word_index
    # idx2word = {v: k for k, v in word2idx.items()}
    # word2idx["PAD"] = 0
    # idx2word[0] = "PAD"
    # vocab_size = len(word2idx)
    # print("vocab size: {:d}".format(vocab_size))
    #len(X_dict)  # 54376
    #X_dict.items()
    X_seq = tokenizer.texts_to_sequences(X)
    X_padded_seq = pad_sequences(X_seq, padding='post', maxlen=max_len)  # 100)
    #X_padded_seq.shape

    return X_padded_seq, YY
def train_val_spliter(faeturs, labels , val_split =0.1):
    #x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, shuffle=True,stratify=True)
    train_idxs, val_idxs= stratified_split(df, target, val_percent=0.2)
    print(x_train[0])
    print(y_train[0])
    return (x_train, y_train),(x_val, y_val)




db = pd.read_csv(data_dir)
#print(len(db)) #4657
#print(db.columns) #['index', 'title', 'genre', 'summary']

#############################################


###################################################

####################################################
# based on tokeizer from tensorflow
# all_sentences = []
# for i in range(len(db)):
#     d = db['summary'][i]
#     all_sentences.append(d)

# tokenizer=Tokenizer()
# tokenizer.fit_on_texts(all_sentences)
# word_dict=tokenizer.word_index
# print(word_dict)
# # convering sequnce of words to sequence of numbers
# seq=tokenizer.texts_to_sequences(all_sentences)
# # padding to have same length sequeces

# padded_seq=pad_sequences(seq,padding='post')
###################################
# text preprocessing


tokenizer=Tokenizer(num_words=10000, oov_token='xxxxxxx')
tokenizer.fit_on_texts(X)
X_dict=tokenizer.word_index
len(X_dict)  # 54376
X_dict.items()
X_seq = tokenizer.texts_to_sequences(X)
X_padded_seq=pad_sequences(X_seq,padding='post',maxlen=max_len)#100)
X_padded_seq.shape
num_records = len(X_padded_seq)
max_seqlen = len(X_padded_seq[0])
print("{:d} sentences, max length: {:d}".format(
num_records, max_seqlen))




from numpy import percentile
quartiles = percentile(data, [25, 50, 75])





# def tokenizer(desc, vocab_dict, max_desc_length):
#     '''
#     Function to tokenize descriptions
#     Inputs:
#     - desc, description
#     - vocab_dict, dictionary mapping words to their corresponding tokens
#     - max_desc_length, used for pre-padding the descriptions where the no. of words is less than this number
#     Returns:
#     List of length max_desc_length, pre-padded with zeroes if the desc length was less than max_desc_length
#     '''
#     a=[vocab_dict[i] if i in vocab_dict else 0 for i in desc.split()]
#     b=[0] * max_desc_length
#     if len(a)<max_desc_length:
#         return np.asarray(b[:max_desc_length-len(a)]+a).squeeze()
#     else:
#         return np.asarray(a[:max_desc_length]).squeeze()
#
# clean_test['desc_tokens']=clean_test['clean_desc'].apply(tokenizer, args=(vocab_dict, max_len))




def create_tf_data(db):
    #targets = db.pop('genre')

    features = db['summary']
    # for ex in ds.take(3):
    #     print(ex[0])
    #     print(ex[1])
    unique_labels = db['genre'].unique()
    le = preprocessing.LabelEncoder()
    le.fit(unique_labels)
    list(le.classes_)
    le.transform(unique_labels)
    label_encode =[]
    for i in range(len(db)):
        label_id = le.transform([db['genre'][i]])
        label_encode.append(label_id.tolist())
    Y = label_encode
    YY = tf.keras.utils.to_categorical(Y, num_classes=10, dtype='float32')

    X = features.apply(clean_description)
    '''
    tokenizer does the heavy lifting in the background, by taking care of
    duplicate/similar words, removing punctuation, and converting letters
    to lower case
    '''
    tokenizer = Tokenizer(num_words=10000, oov_token='xxxxxxx')
    tokenizer.fit_on_texts(X)
    X_dict = tokenizer.word_index
    #len(X_dict)  # 54376
    #X_dict.items()
    X_seq = tokenizer.texts_to_sequences(X)
    X_padded_seq = pad_sequences(X_seq, padding='post', maxlen=200)  # 100)
    ds = tf.data.Dataset.from_tensor_slices((X_padded_seq, YY))
    ds = ds.shuffle(len(db), reshuffle_each_iteration=False)
    data = ds.padded_batch(32, padded_shapes=([-1], []))
    ds_train = data.take(np.int(0.1*len(db)))
    ds_valid = data.skip(np.int(0.1*len(db)))
    # test_dataset = dataset.take(test_size)
    # val_dataset = dataset.skip(test_size).take(val_size)
    # train_dataset = dataset.skip(test_size + val_size)
    # BATCH_SIZE = 128
    # test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
    # val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
    # train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
    return ds_train , ds_valid







