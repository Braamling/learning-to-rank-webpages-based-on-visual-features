import torch
import logging
import torch.nn as nn
from torch.autograd import Variable

logger = logging.getLogger("ViP")


class ViP_features(nn.Module):
    def __init__(self, region_height, feature_size, batch_size):
        super(ViP_features, self).__init__()
        self.use_gpu = torch.cuda.is_available()
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.region_height = region_height
        self.local_perception_layer = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.hidden_dim = 10
        self.lstm = nn.LSTM(256, self.hidden_dim)
        self.reldecision = nn.Linear(self.hidden_dim, self.feature_size)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.use_gpu:
            return (Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                    Variable(torch.zeros(1, batch_size, self.hidden_dim)))

    def apply_lstm(self, x):
        for i, layer in enumerate(x):
            if self.use_gpu:
                layer = layer.cuda()
            else:
                layer = layer

            if i is 0:
                batch_size = layer.size()[0]
                hidden = self.init_hidden(batch_size)

            layer = self.local_perception_layer(layer)
            out, hidden = self.lstm(layer.view(1, -1, 256), hidden)

        return out.squeeze(0)

    def forward(self, x):
        height = x.size()[2]
        splits = int(height / self.region_height)
        x = torch.split(x, splits, 2)

        lstm_out = self.apply_lstm(x)
        tag_space = self.reldecision(lstm_out)
        return tag_space
