import tensorflow as tf
import numpy as np
from ConFig.config import ConfigReader
from modules.datareader import DataLoader
from modules.model import GenreModel
from modules.evaluation import calc_confusion_matrix, calc_precision_recall_f1, calc_loss
tf.random.set_seed(1)
#from modules.
#from modules.callbacks import callbacks
def get_config_data_model():
    cfg = ConfigReader()
    data_train, data_validation = DataLoader()
    model = GenreModel()
    return cfg, data_train , data_validation , model
cfg, data_train , data_validation , model = get_config_data_model()
model.build(input_shape=(None, None, 5))
total_data  = len(data_train[0])

#num_val = 0.2*total_data
#idx = np.random.rand(size = num_val , total_data)
#df_val = df[idx]
#df_train = df.drop[idx]
total_step_train = len(data_train[0]/cfg.batchSize)
def train_one_epoch():
    for step in total_step_train:
        x, y =  data_train[0][step*cfg.batchSize :(step+1)*cfg.batchSize], data_train[1][step*cfg.batchSize :(step+1)*cfg.batchSize]
        with tf.GradientTape as tape:
            logits = model(x)
            lossBatch = calc_loss(y , logits)
            #accBatch = calc_acc(y, logits)
            cm = calc_confusion_matrix(y, logits)
            p , r, f = calc_precision_recall_f1(y, logits)
        grads = tape.gradient(lossBatch, model.trainable_variables)
        model.optimizer.apply_gradient(zip(grads, model.trainable_variables))
        #train_acc_metric.update_state(y, logits)
        if step%200 ==0 :
            print('Training loss(for one  batch) at step %d:%.4f'%(step, float(lossBatch)))
  #  train_acc = train_acc_metric.result()
  #  train_acc_metric.reset_states()
    return loss, acc
total_step_valid = len(data_validation/cfg.batchSize)
def val_one_epoch():
    for step in total_step_valid:
        x, y = data_validation[0][step*cfg.batchSize :(step+1)*cfg.batchSize], data_validation[1][step*cfg.batchSize :(step+1)*cfg.batchSize]
        logits = model(x)
        lass = calc_loss(y, logits)
        cm = calc_confusion_matrix(y, logits)
        p, r, f = calc_precision_recall_f1(y, logits)
 #       val_acc_metric.update_state(y, val_logits)
 #   val_acc = val_acc_metric.result()
 #   val_acc_metric.reset_states()
    return val_loss, val_acc

from modules.callbacks import earlystopping, lr_scheduler, tensorboard_cb_ctc , checkpoint_ctc, backup_ckpt_ctc
_callbacks = [earlystopping, lr_scheduler , tensorboard_cb_ctc, checkpoint_ctc,backup_ckpt_ctc ]
callbacks = tf.keras.callbacks.CallbackList(_callbacks, add_history=True, model=model)
logs={}

for epoch in range(cfg.TotalEpoch):
    callbacks.on_epoch_begin(epoch, logs=logs)
    loss , acc  = train_one_epoch()
    val_loss , val_acc = val_one_epoch()



def keras_model_training(text_model, X_padded_seq, YY):
    text_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # change loss and acc
    text_model.summary()
    text_model.fit(X_padded_seq, YY, epochs=cfg.TotalEpoch, batch_size=cfg.batchSize)


