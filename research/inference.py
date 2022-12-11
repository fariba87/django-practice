from train import model
def inference():
    load_model
    predict on new data



def bookLSTM(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.name="Book Model2"
    model.add(Embedding(len(params['vocab'])+1, output_dim=x_train.shape[1], input_length=x_train.shape[1]))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(200))
    model.add(Dense(1, activation=params['activation']))
    model.compile(loss=params['loss'],
              optimizer=params['optimizer'],
              metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train,
          y_train,
          validation_data=(x_val, y_val),
          batch_size=params['batch_size'],
          epochs=params['epochs'])
    results = model.evaluate(x_test, y_test, batch_size=params['eval_batch_size'])
    return model
def reviewBook(model,text):
  labels = [‘fiction’, ‘nonfiction’]
  a = clean_text(fantasy)
  a = tokenizer(a, vocab_dict, max_desc_length)
  a = np.reshape(a, (1,max_desc_length))
  output = model.predict(a, batch_size=1)
  score = (output>0.5)*1
  pred = score.item()
  return labels[pred]

#
# _, sample_idxs = stratified_split(clean_book, 'label', 0.1)
# train_idxs, val_idxs = stratified_split(clean_book, 'label', val_percent=0.2)
# sample_train_idxs, sample_val_idxs = stratified_split(clean_book[clean_book.index.isin(sample_idxs)], 'label', val_percent=0.2)
# classes=list(clean_book.label.unique())
# sampling=False
# x_train=np.stack(clean_book[clean_book.index.isin(sample_train_idxs if sampling else train_idxs)]['desc_tokens'])
# y_train=clean_book[clean_book.index.isin(sample_train_idxs if sampling else train_idxs)]['label'].apply(lambda x:classes.index(x))
# x_val=np.stack(clean_book[clean_book.index.isin(sample_val_idxs if sampling else val_idxs)]['desc_tokens'])
# y_val=clean_book[clean_book.index.isin(sample_val_idxs if sampling else val_idxs)]['label'].apply(lambda x:classes.index(x))
# x_test=np.stack(clean_test['desc_tokens'])
# y_test=clean_test['label'].apply(lambda x:classes.index(x))