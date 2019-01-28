from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import random
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from .featureStorage import FeatureStorage
from .customExceptions import NoRelDocumentsException

logger = logging.getLogger('Dataset')

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


class ClueWeb12Dataset(Dataset):
    def __init__(self, image_dir=None, features_file=None, get_images=False, query_specific=False, only_with_image=False,
                 size=(64, 64), grayscale=True, vector_cache=None, saliency_dir=None, saliency_cache=None):
        """
        Args:
            img_dir (string): directory containing all images for the ClueWeb12 webpages
            features_file (string): a file containing the features scores for each query document pair.
            scores_file (string): a file containing the scores for each query document pair.
        """
        self.get_images = get_images
        self.query_specific = query_specific
        self.only_with_image = only_with_image
        self.image_dir = image_dir
        self.vector_cache = vector_cache
        self.cache = vector_cache is not None
        self.saliency_cache = saliency_cache
        self.saliency_dir = saliency_dir

        # External doc id to internal doc id
        self.ext2int = {}
        self.idx2posneg = {}

        self.make_dataset(image_dir, features_file)
        if grayscale:
            self.img_transform = transforms.Compose([transforms.Resize(size, interpolation=2),
                                                     transforms.Grayscale(),
                                                     transforms.ToTensor()])
        else:
            # Normalize the input images for torchvision pretrained models:
            # http://pytorch.org/docs/master/torchvision/models.html
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.img_transform = transforms.Compose([transforms.Resize(size, interpolation=2),
                                                     transforms.ToTensor(),
                                                     normalize])

        self.saliency_transform = transforms.Compose([transforms.Resize((64,64), interpolation=2),
                                                 transforms.Grayscale(),
                                                 transforms.ToTensor()])

    def make_dataset(self, image_dir, features_file):
        feature_storage = FeatureStorage(features_file, image_dir, self.query_specific, self.only_with_image,
                                         vector_cache=self.vector_cache, saliency_dir=self.saliency_dir,
                                         saliency_cache=self.saliency_cache)
        dataset = []

        i = 0
        # Create a dataset with all query-document pairs
        for q_id, score, d_id, vec, image, saliency in feature_storage.get_all_entries():

            # Make the query-score index.
            qs_idx = "{}:{}".format(q_id, score)

            # Create an entry for a query-document apir
            query_doc_idx = "{}:{}".format(q_id, d_id)

            # Add an entry with all documents that have a different score
            if qs_idx not in self.idx2posneg:
                posnegs = self._get_alt_scores_docs(feature_storage, q_id, score)
                self.idx2posneg[qs_idx] = posnegs

            # Check whether any documents were found with a different relevant score 
            # and if we filter all documents without an images whether the images exists.
            if len(self.idx2posneg[qs_idx]) > 0:
                self.ext2int[query_doc_idx] = i
                i += 1

                # Create the dataset entry
                item = (image, q_id, score, d_id, vec, saliency)
                dataset.append(item)

        logger.info("Added a dataset with {} queries with {} documents".format(len(feature_storage.get_queries()), i))

        # Convert all external ids in idx2posneg to internal ids.
        for qs_idx in self.idx2posneg.keys():
            self.idx2posneg[qs_idx] = [self.ext2int[i] for i in self.idx2posneg[qs_idx] if
                                       len(self.idx2posneg[qs_idx]) > 0]

        self.dataset = dataset

    """
    Get all documents within the same query with a different score.

    featureStorage: FeatureStorage, a loaded feature storage to retrieve query-docs pairs from
    q_id: int, the query id to search documents in
    score: int, the score of the current document, only documents with other scores are added. 
    """

    def _get_alt_scores_docs(self, featureStorage, query_id, doc_score):
        ids = []
        for score, doc_id in featureStorage.get_documents_in_query(query_id):
            if doc_score != score:
                ids.append("{}:{}".format(query_id, doc_id))

        return ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, q_id, score, _, _, _ = self.dataset[idx]

        # Get the id for the query-score pair
        qs_idx = "{}:{}".format(q_id, score)

        posneg_idx = random.randint(1, len(self.idx2posneg[qs_idx]))
        posneg_idx = self.idx2posneg[qs_idx][posneg_idx - 1]

        if self.dataset[idx][2] > self.dataset[posneg_idx][2]:
            p_image, _, p_score, _, p_vec, p_saliency = self.dataset[idx]
            n_image, _, n_score, _, n_vec, n_saliency = self.dataset[posneg_idx]
        else:
            p_image, _, p_score, _, p_vec, p_saliency = self.dataset[posneg_idx]
            n_image, _, n_score, _, n_vec, n_saliency = self.dataset[idx]

        # Load the positive and negative input image
        if self.get_images:
            p_image = self._load_image(p_image)
            n_image = self._load_image(n_image)

        if self.saliency_dir:
            p_saliency = self._load_saliency_image(p_saliency)
            n_saliency = self._load_saliency_image(n_saliency)

        # The model will filter out the vector, but an empty vector is not supported by pytorch.
        if len(n_vec) is 0:
            p_vec = -1
            n_vec = -1

        positive_sample = (p_image, p_vec, p_score, p_saliency)
        negative_sample = (n_image, n_vec, n_score, n_saliency)

        return positive_sample, negative_sample

    """
    Get a specific Clueweb document
    """

    def get_document(self, doc_id, query_id):
        query_doc_idx = "{}:{}".format(query_id, doc_id)

        if query_doc_idx not in self.ext2int:
            print(query_doc_idx)
            raise NoRelDocumentsException("document not in index, probably no relevant documents were found")

        image, _, score, _, vec, saliency = self.dataset[self.ext2int[query_doc_idx]]

        # The model will filter out the vector, but an empty vector is not supported by pytorch.
        if len(vec) is 0:
            vec = -1

        if self.get_images:
            image = self._load_image(image)

        if self.saliency_dir:
            saliency = self._load_saliency_image(saliency)
        return image, vec, score, saliency

    def _load_image(self, image):
        if self.cache:
            return torch.Tensor(self.vector_cache[image])

        return self.img_transform(default_loader(image))

    def _load_saliency_image(self, image):
        if self.saliency_cache is not None:
            return torch.Tensor(self.saliency_cache[image])

        return self.saliency_transform(default_loader(image))