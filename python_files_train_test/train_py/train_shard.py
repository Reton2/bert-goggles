import sys

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import itertools
import numpy as np

import tensorflow as tf

from keras.optimizers import Adam

from bert_helper_se import create_bert, BertSquadError

from datetime import datetime

MDL_SZE = str(sys.argv[1])
LR = float(sys.argv[2])
BS = int(sys.argv[3])

BERT_FOLDER = "models/bert_{}".format(MDL_SZE)
DATA_FOLDER = 'squad_features_se/{}_350/'.format(MDL_SZE)
VAL_FOLDER = 'squad_features_se/{}_350/'.format(MDL_SZE)
VOCAB_FILE = 'vocab.txt'

options = [(LR, BS)]#, (1e-3, 128)]

for learning_rate, batch_size in options:

   strategy = tf.distribute.MultiWorkerMirroredStrategy()
   options = tf.data.Options()
   options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

   # create model
   print()
   print('Initiating Model, learning rate set to {}, batch size set to {}'.format(learning_rate, batch_size))
   optim = Adam(lr=learning_rate, decay=0.01)
   maxlen = 350
   with strategy.scope():
      bert_model = create_bert(BERT_FOLDER, optim, maxlen, train_to=-55)

   # MODEL SAVE PATHS
   CHKPT_SAVE_DIR = 'chkpts/{}_chkpts/bert_{}_se_'.format(MDL_SZE, MDL_SZE) + str(learning_rate) + '_' + str(maxlen) + '_' + str(batch_size)+ '_' + str(datetime.now()).replace(' ', '_') + '/'

   # LOAD data
   print()
   print(CHKPT_SAVE_DIR)
   print()
   print('Loading training data max_len={}'.format(maxlen))
   train = np.load(DATA_FOLDER + 'squad_feats_se.npy')
   labels = np.load(DATA_FOLDER + 'squad_labs_se.npy')

   print()
   print('Loading validation Data')
   train_val = np.load(VAL_FOLDER + 'squad_feats_se_val.npy')
   labels_val = np.load(VAL_FOLDER + 'squad_labs_se_val.npy')

   # as tensors
   train_data = [tf.constant(train[0]), tf.constant(
       train[1]), tf.constant(train[2])]
   train_labels = tf.constant(labels)

   dataset = tf.data.Dataset.from_tensor_slices(({"Input-Token": train_data[0],
                                   "Input-Segment": train_data[1],
                                   "Input-Masked": train_data[2]},
                                  train_labels))
   dataset = dataset.batch(batch_size).with_options(options)

#   val_data = [tf.constant(train_val[0]), 
#               tf.constant(train_val[1]), 
#               tf.constant(train_val[2])]
#   val_labels = tf.constant(labels_val)

   #val_dataset = tf.data.Dataset.from_tensor_slices(({"Input-Token": val_data[0],
   #                                "Input-Segment": val_data[1],
   #                                "Input-Masked": val_data[2]},
   #                               val_labels))
   #val_dataset = val_dataset.with_options(options)

   if len(tf.config.list_physical_devices()) == 1:
      raise Exception('GPU NOT FOUND')

   print()
   print('Training model')
   # train
   bert_model.fit(dataset,
#                  validation_data=(val_data, val_labels),
                  verbose=1, epochs=4)

#   scores = bert_model.predict(train_data)

#  error = BertSquadError()(train_labels, scores)

   print()
#   print(error)
   print('##################################################################################################')

   # save model
   bert_model.save_weights(CHKPT_SAVE_DIR )

