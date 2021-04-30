from bert import Bert
from answerini import Answerini
from logistic import Logistic
import pickle as pkl

class BertGoggles(object):

    def __init__(self, bert_dir, bert_chkpt_dir,
                 bert_maxlen, answerini_index, logistic_dir=None, top_n=10):

        self.bert_model = Bert(bert_maxlen, bert_dir, bert_chkpt_dir)
        self.answerini = Answerini(answerini_index)
        self.top_n = top_n

        if logistic_dir is not None:
            with open(logistic_dir, 'rb') as file:
                self.logistic = pkl.load(file)
        else:
            self.logistic = None

    def answer(self, question):

        answers = self.answerini.search(question, top_n=self.top_n)

        bert_answers = self.bert_model.search(question, answers)

        return bert_answers, answers
