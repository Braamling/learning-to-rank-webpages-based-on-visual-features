# Transform the saliency input to a vector using conv and dense layers
import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("saliency_conv")


class SaliencyConv(nn.Module):
    def __init__(self):
        super(SaliencyConv, self).__init__()

        self.feature_size = 10

        self.conv1 = nn.Conv2d(1, 2, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 4, 4)
        self.fc1 = nn.Linear(4 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.feature_size)

        # self._initialize_weights()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

        return x
