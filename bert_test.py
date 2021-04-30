import os
import re

import numpy as np

import tensorflow as tf

from official.nlp import bert
import official.nlp.bert.configs
import official.nlp.bert.tokenization

from keras_bert.loader import load_trained_model_from_checkpoint

import keras
from keras.models import Model
from keras.layers import Dense

from settings import LOWER_CASE, VOCAB_FILE

class Bert():

    def __init__(self, maxlen, bert_dir, bert_ckpt_dir):
        # Set up tokenizer to generate Tensorflow dataset
        self.sequence_length = maxlen

        self.tokenizer = self._create_tokenizer(bert_dir)

        self.bert_model = self._create_model(bert_dir, bert_ckpt_dir)

    def _create_model(self, bert_dir, bert_ckpt_dir):

        config_file = os.path.join(bert_dir, 'bert_config.json')
        checkpoint_file = os.path.join(bert_dir, 'bert_model.ckpt')
        bert_pretrained = load_trained_model_from_checkpoint(
            config_file, checkpoint_file, training=True,
            seq_len=self.sequence_length)

        sequence_output = bert_pretrained.layers[-7].output
        pool_output = pool_output = Dense(
            2, name='real_output')(sequence_output)

        bert_model = Model(inputs=bert_pretrained.input, outputs=pool_output)
        bert_model.compile()           
        bert_model.load_weights(bert_ckpt_dir).expect_partial()
        
        return bert_model

    def _create_tokenizer(self, bert_dir):

        return bert.tokenization.FullTokenizer(
            vocab_file=os.path.join(bert_dir, VOCAB_FILE),
            do_lower_case=LOWER_CASE)

    def _get_bert_ans(self, bert_results, orig_to_tokens, content):

        # given bert_results and orig_to_tokens are a one element list
        orig_to_tokens = orig_to_tokens

        start_token_index = bert_results[0][0]
        end_token_index = bert_results[0][1]

        start = orig_to_tokens[start_token_index]
        end = orig_to_tokens[end_token_index]

        ans = ' '.join(np.asarray(re.split('[\\s]+', content))[start:end + 1])

        return ans

    def encode_contents(self, ques, contents):

        encoded_contents = []
        all_data = []
        current = 0

        for q, c in zip(ques, contents):
            encoded = {}

            encoded_paras, offsets, start_index, ends = \
                self.encode_ques(q, c)

            all_data.extend(encoded_paras)

            encoded['range'] = current, current + len(encoded_paras)
            encoded['offsets'] = offsets
            encoded['start_index'] = start_index
            encoded['ends'] = ends

            current += len(encoded_paras)

            encoded_contents.append(encoded)

        all_data = tf.constant(all_data)

        all_data = tf.transpose(all_data, [1, 0, 2])

        return [all_data[0], all_data[1], all_data[2]], encoded_contents

    def predict(self, ques, contents, data, indices):

        # data, indices = self._encode_contents(ques, contents)

        all_log_softmax = self.bert_model.predict(data)

        results = []

        for content, indice in zip(contents, indices):

            ls_range = indice['range']
            offsets = indice['offsets']
            start_index = indice['start_index']
            ends = indice['ends']

            log_softmax = all_log_softmax[ls_range[0]:ls_range[1]]

            indices_scores = [self._score_index(
                ls, start_index, end) for ls, end in zip(log_softmax, ends)]

            best_index = np.argmax([i_s[1] for i_s in indices_scores])

            best_indices = indices_scores[best_index][0] + offsets[best_index]

            best_scores = indices_scores[best_index][1]

            s_null = indices_scores[best_index][2]

            result = [best_indices.astype(int), best_scores, s_null]

            tok_to_origs = self._content_token_map(content)

            ans_text = self._get_bert_ans(result, tok_to_origs, content)

            result[0] = ans_text

            results.append(result)

        return results

    def _content_token_map(self, content):
        tok_to_orig = []

        for i, word in enumerate(re.split('[\\s]+', content)):
            for j in self.tokenizer.tokenize(word):
                tok_to_orig.append(i)

        tok_to_orig = np.asarray(tok_to_orig)

        return tok_to_orig

    def _score_index(self, bert_result, start_index, end):
        start_scores = bert_result[:, 0]
        end_scores = bert_result[:, 1]

        highest_score = -float('inf')
        best_indices = np.zeros(2)
        for i, start_score in enumerate(start_scores[:end + 1]):
            if i < start_index:
                continue
            for j, end_score in enumerate(end_scores[i:end + 1]):
                sequence_score = start_score + end_score
                if sequence_score > highest_score:
                    best_indices[0] = i - start_index
                    best_indices[1] = j + i - start_index
                    highest_score = sequence_score
        s_null = start_scores[0] + end_scores[0]
        return best_indices, highest_score, s_null

    def encode_ques(self, ques, context):

        parts = []
        offsets = [0]

        text = re.split('[\\s]+', context)
        len_left = len(self.tokenizer.tokenize(context))
        max_length = self.sequence_length - \
            len(self.tokenizer.tokenize(ques)) - 3

        index = 0
        while len_left > max_length:
            part = []
            count = 0
            index_covered = 0
            for w in text[index:]:

                sub_tokens_num = len(self.tokenizer.tokenize(w))
                if count + sub_tokens_num < max_length:

                    part.append(w)
                    len_left -= sub_tokens_num
                    count += sub_tokens_num
                    index_covered += 1

            index += index_covered
            offsets.append(index)
            parts.append(' '.join(part))

        parts.append(' '.join(text[index:]))

        results = []
        ends = []
        start = 0
        for part in parts:
            r, start, end = self._bert_encode(ques, part)
            ends.append(end)
            results.append(r)

        return results, offsets, start, ends

    def _bert_encode(self, ques, context):

        tokens = []
        input_types = []
        input_mask = []

        max_len = self.sequence_length

        tokens.append('[CLS]')
        input_types.append(0)
        input_mask.append(1)

        for token in self.tokenizer.tokenize(ques):
            tokens.append(token)
            input_types.append(0)
            input_mask.append(1)

        tokens.append('[SEP]')
        input_types.append(0)
        input_mask.append(1)

        content_start = len(input_mask)

        for token in self.tokenizer.tokenize(context):
            tokens.append(token)
            input_types.append(1)
            input_mask.append(1)

        content_end = len(input_mask) - content_start

        if len(tokens) > max_len - 1:
            tokens = tokens[:max_len - 1]
            input_types = input_types[:max_len - 1]
            input_mask = input_mask[:max_len - 1]

        tokens.append('[SEP]')
        input_types.append(1)
        input_mask.append(1)

        while len(tokens) < max_len:
            tokens.append('[PAD]')
            input_types.append(0)
            input_mask.append(0)

        input_ids = np.asarray(self.tokenizer.convert_tokens_to_ids(tokens))
        input_types = np.asarray(input_types)
        input_mask = np.asarray(input_mask)
        # input_mask = self.mask_list(tokens)

        return ([input_ids,
                 input_types,
                 input_mask], content_start, content_end)
