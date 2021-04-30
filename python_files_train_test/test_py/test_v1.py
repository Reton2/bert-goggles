import pickle as pkl

import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from bert_tester_v1 import Bert, test_v1

import numpy as np

MDL_SZE = str(sys.argv[1])

print('This file contains the Validation scores for all BERT {} models'.format(MDL_SZE))
print('Test is based on v1.1 test')

CHKPT_DIR = 'chkpts/' + MDL_SZE + '_chkpts'
MODEL_DIR = 'models/bert_' + MDL_SZE
CONTEXT_FILE = 'squad_validation/contexts_v2.pkl'
COL_FILE = 'squad_validation/collection_v2.pkl'
MAXLEN = 350

# load pickles
print()
print('Loading pickles')
with open(CONTEXT_FILE, 'rb') as f:
   docs = pkl.load(f)

with open(COL_FILE, 'rb') as f:
   collection = pkl.load(f)

collection = collection[:len(collection)//4]

chkpt_list = os.listdir(CHKPT_DIR)

dirs = [os.path.join(CHKPT_DIR, chkpt, '') for chkpt in chkpt_list if 'bert_' + MDL_SZE in chkpt]

for directory in dirs:
   print(directory)
   # load bert model
   print()
   print('Loading Bert Model')
   bert_model = Bert(MAXLEN, MODEL_DIR, directory)

   # testing
   print()
   print('Testing Model')
   em, f1, recall = test_v1(bert_model, collection, docs)

   em = np.asarray(em)
   f1 = np.asarray(f1)
   recall = np.asarray(recall)

   # score
   f1_score_calc = f1.mean()
   acc_score_calc = em.mean()
   recall_score_calc = recall.mean()

   print('Tesing model at: {}'.format(directory))
   print('F1 Score achieved: {:.3f}'.format(round(f1_score_calc, 5) * 100))
   print('Recall Score achieved: {:.3f}'.format(round(recall_score_calc, 5) * 100))
   print('EM Score achieved: {:.3f}'.format(round(acc_score_calc, 5) * 100))

   print('#################################################################')
   print()
   
