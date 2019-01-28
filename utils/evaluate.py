import logging
import time

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

from .featureStorage import FeatureStorage
from .customExceptions import NoImageAvailableException, NoRelDocumentsException

logger = logging.getLogger('Evaluate')

"""
This class can be used to easily evaluate a LTR model in various 
stages of the training process.

Evaluation measures are taken from https://gist.github.com/bwhite/3726239
"""


class Evaluate():
    def __init__(self, path, dataset, load_images, prefix, vector_cache=None, batch_size=None):
        self.dataset = dataset
        self.storage = FeatureStorage(path, dataset.image_dir, dataset.query_specific, dataset.only_with_image,
                                      vector_cache=vector_cache)
        self.prepare_eval_data()
        self.use_gpu = torch.cuda.is_available()
        self.load_images = load_images
        self.prefix = prefix
        self.batch_size = batch_size

    """
    Get all the ranked queries and their scores. 
    """

    def prepare_eval_data(self):
        self.query_ids = self.storage.get_queries()
        self.queries = {}
        for q_id in self.query_ids:
            self.queries[q_id] = self.storage.get_scores(q_id)

    def _eval_query(self, query_id, model, get_df):
        try:
            predictions, ranked_docs = self._get_scores(query_id, model)

            scores = Evaluate.compute_scores(predictions)

            if get_df:
                self.add_to_df(query_id, ranked_docs, predictions)

        except NoRelDocumentsException as e:
            logger.warning("query {} gave an exception: {}".format(query_id, e))
            self.failed += 1
            return {}
        except Exception as e:
            logger.error("Throwing the error from loading a documet in query {}...".format(query_id))
            raise e

        return scores

    def _get_predictions(self, model, doc_ids, scores, query_id):
        batch_vec = []
        images = []
        batch_saliency = []
        for doc, score in zip(doc_ids, scores):
            try:
                image, vec, rel_score, saliency = self.dataset.get_document(doc, query_id)

                batch_vec.append(vec)

                if saliency is not False:
                    batch_saliency.append(saliency)
                images.append(image)

                if score is not rel_score:
                    logger.error(query_id, doc, score, rel_score, vec)
                    raise Exception("Somehow the relevance score in the dataset and feature storage are different.")
            except NoImageAvailableException as e:
                logger.debug("Document {} in query {} does not have an image and is excluded from evaluation. ".format(doc, query_id))
            except IOError as e:
                logger.error("Document {} in query {} gave an exception while loading, file is probabily corrupt. ".format(doc, query_id))
                raise e

        batch_vec = np.vstack(batch_vec)

        if len(batch_saliency) > 0:
            batch_saliency = torch.stack(batch_saliency)
        else:
            batch_saliency = None

        if self.load_images:
            images = torch.stack(images)

        if self.use_gpu:
            batch_vec = Variable(torch.from_numpy(batch_vec).float().cuda())

            if batch_saliency is not None:
                batch_saliency = Variable(batch_saliency.float().cuda())

            if self.load_images:
                images = Variable(images.float().cuda())
        else:
            batch_vec = Variable(torch.from_numpy(batch_vec).float())

            if batch_saliency is not None:
                batch_saliency = Variable(batch_saliency.float())
            if self.load_images:
                images = Variable(images.float())

        if batch_saliency is not None:
            batch_pred = model.forward(images, batch_vec, batch_saliency).data.cpu().numpy()
        else:
            batch_pred = model.forward(images, batch_vec).data.cpu().numpy()

        return list(batch_pred.flatten())

    def _get_scores(self, query_id, model):
        logger.debug('Starting to prepare {} batch to evaluate query {}'.format(self.prefix, query_id))
        start = time.time()

        predictions = []
        scores = []
        doc_ids = []
        batch_scores = []
        batch_doc_ids = []
        i = 0
        for doc, score in self.queries[query_id]:
            if self.batch_size is not None and i >= self.batch_size:
                predictions += self._get_predictions(model, batch_doc_ids, batch_scores, query_id)
                i = 0
                batch_scores = []
                batch_doc_ids = []
            else:
                i += 1
                scores.append(score)
                batch_scores.append(score)
                doc_ids.append(doc)
                batch_doc_ids.append(doc)

        if i > 0:
            predictions += self._get_predictions(model, batch_doc_ids, batch_scores, query_id)

        predictions = [(pred, score, doc_ids) for pred, score, doc_ids in zip(predictions, scores, doc_ids)]

        # Shuffle the prediction before sorting to make sure equal predictions are
        # in random order.
        np.random.shuffle(predictions)
        predictions = sorted(predictions, key=lambda x: -x[0])
        _, predictions, doc_ids = zip(*predictions)

        logger.debug('Sorted predictions, {} seconds since start'.format(time.time() - start))
        return predictions, doc_ids

    def add_scores(self, scores, to_add_scores):
        for key in to_add_scores.keys():
            if key not in scores:
                scores[key] = 0
            scores[key] += to_add_scores[key]

        return scores

    def avg_scores(self, scores, n):
        for key in scores.keys():
            scores[key] = scores[key] / n

        return scores

    def print_scores(self, scores):
        for key in sorted(list(scores.keys())):
            logger.info("{}_{} {}".format(self.prefix, key, scores[key]))

    def store_scores(self, path, description, scores, final=False):
        """
        Append a dict of scores to file.

        Path: Full path to the file to be stored
        Description: Prefix for run identification to the scores that will be appended
        Scores: Dict with score name as key and score value as value.
        """
        scores = " ".join(["{0}:{1:.4f}".format(k, scores[k]) for k in sorted(list(scores.keys()))])
        with open(path, "a") as f:
            f.write("{}-{} {}\n".format(self.prefix, description, scores))

        # Write final results to aggregate file with model name prefix
        if final:
            with open('storage/logs/{}_{}'.format(self.prefix, description.split("_")[0]), "a") as f:
                f.write("{} {}\n".format(scores, description))


    def _log_scores(self, scores, tf_logger, epoch):
        for key in scores.keys():
            tf_logger.log_value('{}_{}'.format(self.prefix, key), scores[key], epoch)

    def add_to_df(self, query_id, docs, scores):
        df = pd.DataFrame(np.asarray([docs, scores]).T, columns=[query_id, str(query_id) + "_s"])
        if self.ranking_df is None:
            self.ranking_df = df
        else:
            self.ranking_df = pd.concat([self.ranking_df, df], axis=1)

    def eval(self, model, tf_logger=None, epoch=None, get_df=False):
        self.failed = 0
        scores = {}
        self.ranking_df = None
        for q_id in self.queries.keys():
            eval_scores = self._eval_query(q_id, model, get_df)
            self.add_scores(scores, eval_scores)

        n = len(self.queries.keys()) - self.failed
        scores = self.avg_scores(scores, n)

        self.print_scores(scores)
        # self.print_ranking_summary(rankings)
        if tf_logger is not None:
            self._log_scores(scores, tf_logger, epoch)

        if get_df:
            return scores, self.ranking_df

        return scores

    @staticmethod
    def compute_scores(predictions):
        # Make sure all negative values are put to 0.
        eval_predictions = np.asarray(predictions)
        eval_predictions[eval_predictions < 0] = 0

        scores = {}
        scores["ndcg@1"] = Evaluate.ndcg_at_k(eval_predictions, 1)
        scores["ndcg@5"] = Evaluate.ndcg_at_k(eval_predictions, 5)
        scores["ndcg@10"] = Evaluate.ndcg_at_k(eval_predictions, 10)
        scores["p@1"] = Evaluate.precision_at_k(eval_predictions, 1)
        scores["p@5"] = Evaluate.precision_at_k(eval_predictions, 5)
        scores["p@10"] = Evaluate.precision_at_k(eval_predictions, 10)
        scores["map"] = Evaluate.average_precision(eval_predictions)

        return scores

    @staticmethod
    def dcg_at_k(r, k, method=0):
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method must be 0 or 1.')
        return 0.

    @staticmethod
    def ndcg_at_k(r, k, method=0):
        dcg_max = Evaluate.dcg_at_k(sorted(r, reverse=True), k, method)
        if not dcg_max:
            return 0.
        return Evaluate.dcg_at_k(r, k, method) / dcg_max

    @staticmethod
    def precision_at_k(r, k):
        """Score is precision @ k
        Relevance is binary (nonzero is relevant).
        >>> r = [0, 0, 1]
        >>> precision_at_k(r, 1)
        0.0
        >>> precision_at_k(r, 2)
        0.0
        >>> precision_at_k(r, 3)
        0.33333333333333331
        >>> precision_at_k(r, 4)
        Traceback (most recent call last):
            File "<stdin>", line 1, in ?
        ValueError: Relevance score length < k
        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Precision @ k
        Raises:
            ValueError: len(r) must be >= k
        """
        assert k >= 1
        r = np.asarray(r)[:k] > 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)

    @staticmethod
    def average_precision(r):
        """Score is average precision (area under PR curve)
        Relevance is binary (nonzero is relevant).
        >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
        >>> delta_r = 1. / sum(r)
        >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
        0.7833333333333333
        >>> average_precision(r)
        0.78333333333333333
        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Average precision
        """
        r = np.asarray(r) > 0
        out = [Evaluate.precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
        if not out:
            return 0.
        return np.mean(out)
