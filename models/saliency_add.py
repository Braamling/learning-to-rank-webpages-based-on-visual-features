# Concat the saliency conv output and visual feature extractor conv and transform using a dense layer
import torch
import logging
import torch.nn as nn

from models.dense import Dense

logger = logging.getLogger("model")

"""
The LTR score model can hold a second feature network that can be fed
external features. The model is then trained end-to-end.
"""


class SaliencyAdd(Dense):
    def __init__(self, visual_model, saliency_model, hidden_layers='4096x4096', output_size=1000, dropout=0.2):
        super(SaliencyAdd, self).__init__()

        self.visual_model = visual_model
        self.saliency_model = saliency_model

        input_size = visual_model.feature_size + saliency_model.feature_size

        self.classifier = nn.Sequential(*self._create_layers(input_size, hidden_layers, output_size, dropout))

        self.feature_size = output_size

        self._initialize_weights()


    def forward(self, image, saliency):
        visual_features = self.visual_model(image)
        saliency_features = self.saliency_model(saliency)

        features = torch.cat((visual_features, saliency_features), 1)

        x = self.classifier(features)

        return x

