import torch.nn as nn

from models.dense import Dense


class TransformCache(Dense):

    def __init__(self, input_size=25088, hidden_layers='4096x4096', output_size=1000, init_weights=True, dropout=0.5):
        super(TransformCache, self).__init__()

        self.classifier = nn.Sequential(*self._create_layers(input_size, hidden_layers, output_size, dropout=dropout))

        self.feature_size = output_size
        if init_weights:
            self._initialize_weights()
