import tensorflow as tf
from collections import Counter
import tensorflow_datasets as tfds
tokenizer = tfds.features.text.Tokenizer()
token_counts = Counter()


for example in ds_raw_train:
    tokens = tokenizer.tokenize(example[0].numpy()[0])
    token_counts.update(tokens)
print('Vocab-size:', len(token_counts))

encoder = tfds.features.text.TokenTextEncoder(token_counts)
example_str = 'This is an example!'
print(encoder.encode(example_str))

def encode(text_tensor, label):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label

def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label],
                          Tout=(tf.int64, tf.int64))
ds_train = ds_raw_train.map(encode_map_fn)
ds_valid = ds_raw_valid.map(encode_map_fn)
ds_test = ds_raw_test.map(encode_map_fn)


embedding_dim = 20
vocab_size = len(token_counts) + 2
tf.random.set_seed(1)
## build the model
bi_lstm_model = tf.keras.Sequential([tf.keras.layers.Embedding(input_dim=vocab_size,
                                                               output_dim=embedding_dim,name='embed-layer'),
                                     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, name='lstm-layer'), name='bidir-lstm'),
                                     tf.keras.layers.Dense(64, activation='relu'),
                                     tf.keras.layers.Dense(1, activation='sigmoid')])
bi_lstm_model.summary()
## compile and train:
bi_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),metrics=['accuracy'])
history = bi_lstm_model.fit(train_data,validation_data=valid_data,epochs=10)

## evaluate on the test data
test_results = bi_lstm_model.evaluate(test_data)
print('Test Acc.: {:.2f}%'.format(test_results[1]*100))