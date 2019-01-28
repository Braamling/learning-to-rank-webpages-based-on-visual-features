from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pathlib
from os import listdir
from os.path import isfile, join

import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

logger = logging.getLogger('Dataset')


class ImageDataset(Dataset):
    def __init__(self, image_dir, size=(64, 64)):
        self.image_dir = image_dir

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.img_transform = transforms.Compose([transforms.Resize(size, interpolation=2),
                                                 transforms.ToTensor(),
                                                 normalize])

    def iterate(self):
        for file in [f for f in listdir(self.image_dir) if isfile(join(self.image_dir, f))]:
            yield file, self.img_transform(default_loader(join(self.image_dir, file)))


class VectorCache():

    def __init__(self, cache_path=None, model_name=None, image_type=None, model=None, indentifier="cache"):
        if cache_path is None:
            self.storage_location = 'storage/model_cache'
            pathlib.Path(self.storage_location).mkdir(parents=True, exist_ok=True)

            self.cache_name = "{}-{}-{}".format(model_name, image_type, indentifier)
            self.cache_path = os.path.join(self.storage_location, self.cache_name)
        else:
            self.cache_path = cache_path

        self.model = model

        if os.path.isfile(self.cache_path):
            self.h5 = h5py.File(self.cache_path, 'r')
            self.vectors = self.h5["vectors"]
        else:
            self.h5 = h5py.File(self.cache_path, 'w')
            self.vectors = self.h5.create_group("vectors")

    def add(self, name, input):
        output = np.squeeze(self.model.cache_forward(input.unsqueeze(0)).data.cpu().numpy(), axis=0)
        self.vectors[name] = output

    def add_images_from_folder(self, folder_name, size):
        dataset = ImageDataset(folder_name, size)
        for name, image in dataset.iterate():
            self.add(name, image)

    def __getitem__(self, name):
        return self.vectors[self._prep_name(name)].value

    def exists(self, name):
        return self._prep_name(name) in self.vectors

    def _prep_name(self, name):
        return name.split("/")[-1]