import numpy as np

from sklearn import linear_model, preprocessing
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler

class Logistic():

    def __init__(self):

        self.regression = LogisticRegression()
        # self._scaler = StandardScaler()

    def predict(self, samples):

        # data = self._scaler.transform(samples)
        result = self.regression.predict_proba(samples)
        predictions = np.argmax(result, axis=1)

        return predictions

    def score(self, samples):

        # data = self._scaler.transform(samples)
        result = self.regression.predict_proba(samples)

        return result[:, 0]

    def train(self, trus, fals, s_null=True):

        data, labels = self._clean_data(trus, fals, s_null)

        # fit regression
        self.regression.fit(data, labels)

    def _clean_data(self, trus, fals, s_null):
        # remove outliers
        trus_filtered = self._keep_percentile(trus, s_null)
        fals_filtered = self._keep_percentile(fals, s_null)

        # balance out
        ra = np.arange(len(fals_filtered))
        ra = np.random.permutation(ra)
        fals_fil_rand = fals_filtered[ra[:len(trus_filtered)]]

        # combine
        data = np.concatenate([trus_filtered, fals_fil_rand])
        labels = np.concatenate(
            [np.ones(len(trus_filtered)), -np.ones(len(fals_fil_rand))])

        # randomize
        seed = np.random.permutation(np.arange(len(data)))
        rand_data = data[seed]
        rand_labels = labels[seed]

        return rand_data, rand_labels

    def _keep_percentile(self, scores, s_null, percentile=0.99):

        def range_bool(min_max, scores):
            return np.asarray([s > min_max[0] and s < min_max[1]
                               for s in scores])

        remove_percentile = (1 - percentile) / 2 * 100

        ans_scs_range = [np.percentile(scores[:, 0], remove_percentile),
                         np.percentile(scores[:, 0], 100 - remove_percentile)]

        bert_scs_range = [np.percentile(scores[:, 1], remove_percentile),
                          np.percentile(scores[:, 1], 100 - remove_percentile)]

        ans_scs_bool = range_bool(ans_scs_range, scores[:, 0])
        bert_scs_bool = range_bool(bert_scs_range, scores[:, 1])

        inside_range = np.logical_and(
            ans_scs_bool, bert_scs_bool)

        if s_null:
            bert_null_range = [np.percentile(scores[:, 2], remove_percentile),
                               np.percentile(scores[:, 2],
                                             100 - remove_percentile)]

            bert_null_bool = range_bool(bert_null_range, scores[:, 2])

            inside_range = np.logical_and(inside_range, bert_null_bool)

        return scores[inside_range]
