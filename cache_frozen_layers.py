import argparse
import os

from models.inception import inception_v3
from models.resnet import resnet18, resnet152
from models.vgg16 import vgg16
from utils.vectorCache import VectorCache

models = {
    "vgg16":     vgg16,
    "resnet152": resnet152,
    "resnet18":  resnet18,
    # "inception": inception_v3,
}

image_paths = ["snapshots", "highlights", "saliency_deepgaze"]


def build_cache():
    for model_name, model in models.items():
        for image_type in image_paths:
            vector_cache = VectorCache(model_name, image_type, model(pretrained=True))

            if model_name in ("vgg16", "resnet152", "resnet18"):
                size = (224, 224)
            if model_name in ("inception"):
                size = (299, 299)

            vector_cache.add_images_from_folder(os.path.join(FLAGS.image_folder, image_type), size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_folder', type=str, default='storage/images_224x224/',
                        help='The location of all the images.')

    FLAGS, unparsed = parser.parse_known_args()

    build_cache()
