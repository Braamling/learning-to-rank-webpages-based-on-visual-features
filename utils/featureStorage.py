import numpy as np
import os.path

from .LETORIterator import LETORIterator

"""
This class acts as an interface for the LETOR storage containing all 
query-document pairs with their contextual features and judgment scores.

The interface can both be used to add new queries, documents and features, 
but also to iterate over the content of a LETOR file.
"""
class FeatureStorage():
    def __init__(self, path, image_dir, query_specific=False, only_with_image=False, vector_cache=None, saliency_dir=None, saliency_cache=None):
        self.letorIterator = LETORIterator(path)
        self.pairs = []
        self.scores = {}
        self.queries = {}
        self.q_docs = {}
        self.only_with_image = only_with_image
        self.query_specific = query_specific
        self.image_dir = image_dir
        self.image_cache = vector_cache
        self.saliency_dir  = saliency_dir
        self.saliency_cache = saliency_cache

        self.parse()


    def parse(self):
        for query_id, doc_id, rel_score, features in self.letorIterator.feature_iterator():
            query_id, rel_score = int(query_id), int(rel_score)
            if query_id not in self.queries:
                self.queries[query_id] = {}

            if rel_score not in self.queries[query_id]:
                self.queries[query_id][rel_score] = {}

            features = [float(f) for f in features]
            self.queries[query_id][rel_score][doc_id] = np.asarray(features)


    """
    Retrieve all features for a document and query pair. Both query and non
    query specific features are retrieved.
    
    query_id: int, id of the query
    document_id: str, id of the document
    score: int, judgement score for the document query pair
    """
    def get_query_document_features(self, query_id, doc_id, rel_score):
        return self.queries[query_id][rel_score][doc_id]

    """
    Get all documents in a query in an array with typle values of (score, document_id)
    """
    def get_documents_in_query(self, query_id):
        documents = []
        for rel_score in self.queries[query_id]:
            for doc_id in self.queries[query_id][rel_score]:
                if self._get_image(query_id, doc_id):
                    documents.append((rel_score, doc_id))

        return documents

    def _get_image(self, q_id, d_id):
        if self.query_specific:
            image_path = os.path.join(self.image_dir, "{}-{}.png".format(q_id, d_id))
        else:
            image_path = os.path.join(self.image_dir, "{}.png".format(d_id))

        if (self.only_with_image and self._is_image_available(image_path)) or not self.only_with_image:
            return image_path
        else:
            return False


    def _is_image_available(self, image_path):
        if self.image_cache is not None:
            return self.image_cache.exists(image_path)
        else:
            return os.path.isfile(image_path)

    def _get_saliency(self, d_id):
        if self.saliency_dir is None and self.saliency_cache is None:
            return False

        saliency_path = os.path.join(self.saliency_dir, "{}.png".format(d_id))

        if self.saliency_cache is not None and self.saliency_cache.exists(saliency_path):
            return saliency_path
        elif os.path.isfile(saliency_path):
            return saliency_path

        return False

    """
    Get all available query-document pairs in an array of with tuple values of
    (query_id, score, document_id, feature_vectore)
    """
    def get_all_entries(self):
        for query_id in self.queries:
            for rel_score in self.queries[query_id]:
                for doc_id in self.queries[query_id][rel_score]:
                    vec = self.queries[query_id][rel_score][doc_id]
                    image = self._get_image(query_id, doc_id)
                    saliency = self._get_saliency(doc_id)

                    if image:
                        yield (query_id, rel_score, doc_id, vec, image, saliency)


    """
    Get all available query-document pairs in an array of with tuple values of
    (query_id, score, document_id, feature_vectore)
    """
    def get_queries(self):
        return list(self.queries.keys())


    """
    Get a sorted list of tuples (doc_id, score) for a specific query
    """
    def get_scores(self, query_id):
        if query_id not in self.scores:
            self.scores[query_id] = []

            for scores in self.queries[query_id].keys():
                for doc_id in self.queries[query_id][scores].keys():
                    if self._get_image(query_id, doc_id):
                        self.scores[query_id].append((doc_id, int(scores)))


            self.scores[query_id] = sorted(self.scores[query_id], key=lambda x: -x[1])

        return self.scores[query_id]

    """
    Get a sorted list of tuples (scores, feature_val) for a specific query
    """
    def get_ranked_scores(self, query_id, feature_id):
        if query_id not in self.scores:
            self.scores[query_id] = []

            for rel_score in self.queries[query_id].keys():
                for doc_id in self.queries[query_id][rel_score].keys():
                    if self._get_image(query_id, doc_id):
                        rank_score = float(self.queries[query_id][rel_score][doc_id][feature_id])
                        self.scores[query_id].append((rel_score, rank_score))

            self.scores[query_id] = sorted(self.scores[query_id], key=lambda x: -x[1])

        return self.scores[query_id]

    def print_index(self):
        self.f.visit(self.printname)

    def printname(self, name):
        print(name)