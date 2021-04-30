import os

import numpy as np

import tensorflow as tf
tf.autograph.set_verbosity(1)
from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

import keras

from keras_bert.bert import get_model
from keras_bert.loader import load_trained_model_from_checkpoint
from keras.optimizers import Adam

from keras.layers import Dense, Input, Flatten, concatenate, Dropout, Lambda, Bidirectional, LSTM
from keras.models import Model
import keras.backend as K
import re
import codecs

VOCAB_FILE = 'vocab.txt'
SQUAD_2_0 = 'squad/v2.0'
SQ_2_0_CONFIG = 1
WORD_LIMIT = 512


def mask_list(tokens, prob=0.15):
    length = len(tokens)

    mask = (np.abs(np.random.random_sample(length)) >= 0.15).astype(int)

    mask[tokens == 102] = 1
    mask[tokens == 101] = 1

    return mask


def bert_prepare_data(encoded_data):

    ids_collecter = tf.ragged.constant([[0]])
    masks_collecter = tf.ragged.constant([[0]])
    types_collecter = tf.ragged.constant([[0]])

    for tokens in encoded_data:
        # input tokens
        input_ids = tokens
        ids_collecter = tf.concat([ids_collecter, [input_ids]], axis=0)

        # input masks
        input_mask = mask_list(tokens)
        masks_collecter = tf.concat(
            [masks_collecter, [input_mask]], axis=0)

        # input types
        ques_end = np.where(input_ids == 102)[0][0] + 1
        input_types = np.concatenate(
            [np.zeros_like(input_ids[:ques_end]),
             np.ones_like(input_ids[ques_end:])])
        types_collecter = tf.concat(
            [types_collecter, [input_types]], axis=0)

    input_ids = ids_collecter[1:].to_tensor()
    input_mask = masks_collecter[1:].to_tensor()
    input_types = types_collecter[1:].to_tensor()

    return [input_ids, input_types, input_mask]


def encode_ques(ques, context, tokenizer, max_len):
    tokens = ['[CLS]', *tokenizer.tokenize(ques),
              '[SEP]', *tokenizer.tokenize(context)]
    if len(tokens) > max_len - 1:
        tokens = tokens[:max_len - 1]

    tokens.append('[SEP]')

    return tokenizer.convert_tokens_to_ids(tokens)


def token_encode_squad(tokenizer, squad, size=0):
    questions = []
    labels = []

    for s in squad.as_numpy_iterator():
        if size and len(questions) == size:
            break

        ques = encode_ques(s['question'], s['context'], tokenizer, WORD_LIMIT)
        is_impos = abs(1 - s['is_impossible'])

        questions.append(ques)
        labels.append(is_impos)

    labels = tf.constant(labels)
    questions = tf.ragged.constant(questions)

    return questions, labels


def create_tokenizer(bert_folder):
    has_vocab = VOCAB_FILE in tf.io.gfile.listdir(bert_folder)

    if not has_vocab:
        raise Exception(
            'Path {} does not contain vocab.txt'.format(bert_folder))

    tokenizer = bert.tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_folder, "vocab.txt"),
        do_lower_case=True)

    return tokenizer


def load_squad():
    # load squad dataset
    sq = tfds.question_answering.squad.Squad()
    sq_config = sq.BUILDER_CONFIGS[SQ_2_0_CONFIG]
    sq = tfds.question_answering.squad.Squad(config=sq_config)
    sq_name = sq.name + '/' + sq.BUILDER_CONFIGS[1].name

    # check if squad/v2.0
    if sq_name != SQUAD_2_0:
        raise Exception('Incorrect dataset: {}'.format(sq_name))

    return sq.as_dataset()


def create_bert(bert_pretained_dir, optimizer, maxlen, train_to=-3):

    config_file = os.path.join(bert_pretained_dir, 'bert_config.json')
    checkpoint_file = os.path.join(bert_pretained_dir, 'bert_model.ckpt')
    bert_pretrained = load_trained_model_from_checkpoint(
        config_file, checkpoint_file, training=True, seq_len=maxlen)
    
    sequence_output = bert_pretrained.layers[-7].output
    pool_output = Dense(2,
                       kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                       name='real_output')(sequence_output)
    bert_model = Model(inputs=bert_pretrained.input, outputs=pool_output)
    bert_model.compile(loss=BertSquadError(),
                       optimizer=optimizer)

    return bert_model

class BertSquadError(tf.keras.losses.Loss):

    def call(self, positions, logits):
        
        logits = tf.transpose(logits, [0, 2, 1])
    
        # logits' shape is [2, squence_length]
        start_logits = logits[:, 0]
        end_logits = logits[:, 1]

        one_hot_positions = tf.one_hot(
            positions, depth=logits.shape[2], dtype=tf.float32)
        # one_hot_positions' shape is [2, squence_length]
        start_positions = one_hot_positions[:, 0]
        end_positions = one_hot_positions[:, 1]

        log_probs = tf.nn.log_softmax(start_logits, axis=-1)
        loss_start = -tf.reduce_mean(tf.reduce_sum(start_positions * log_probs, axis=-1),axis=-1)

        log_probs = tf.nn.log_softmax(end_logits, axis=-1)
        loss_end = -tf.reduce_mean(tf.reduce_sum(end_positions * log_probs, axis=-1),axis=-1)

        loss_total = tf.reduce_sum([loss_start, loss_end])

        return loss_total   
