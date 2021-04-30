from pyserini.search import SimpleSearcher
import json
import collections

class Answerini(object):

    def __init__(self, index_dir):

        self.searcher = SimpleSearcher(index_dir)

    def search(self, question, top_n):

        decoder = json.JSONDecoder()

        Answer = collections.namedtuple(
            "Answer", ['id', "score", "content"])

        answers = []

        hits = self.searcher.search(question, k=top_n)

        for hit in hits:
            content = decoder.decode(hit.raw)['contents']
            score = hit.score
            doc_id = hit.docid

            answer = Answer(id=doc_id, score=score, content=content)

            answers.append(answer)

        return answers
