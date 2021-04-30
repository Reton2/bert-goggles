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

BERT_FOLDER = "models/bert_" + MDL_SZE
DATA_FOLDER = 'squad_features_se/{}_350/'.format(MDL_SZE)
VAL_FOLDER = 'squad_features_se/{}_350/'.format(MDL_SZE)
VOCAB_FILE = 'vocab.txt'

options = [(3e-3, 64), (3e-3, 32),
           (1e-3, 64), (1e-3, 32),
           (5e-4, 64), (5e-4, 32)]#, (5e-3, 16), (5e-3, 8)]

for learning_rate, batch_size in options:
   # create model
   print()
   print('Initiating Model, learning rate set to {}, batch size set to {}'.format(learning_rate, batch_size))
   optim = Adam(lr=learning_rate, decay=0.01)
   maxlen = 350
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

   val_data = [tf.constant(train_val[0]), 
               tf.constant(train_val[1]), 
               tf.constant(train_val[2])]
   val_labels = tf.constant(labels_val)

   if len(tf.config.list_physical_devices()) == 1:
      raise Exception('GPU NOT FOUND')

   print()
   print('Training model')
   # train
   bert_model.fit(x=train_data, y=train_labels,
                  validation_data=(val_data, val_labels),
                  verbose=1, batch_size=batch_size, epochs=4)

   scores = bert_model.predict(train_data)

   error = BertSquadError()(train_labels, scores)

   print()
   print(error)
   print('##################################################################################################')

   # save model
   bert_model.save_weights(CHKPT_SAVE_DIR )

