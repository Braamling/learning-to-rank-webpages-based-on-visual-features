import torch
import logging
import torch.nn as nn
logger = logging.getLogger("model")

"""
The LTR score model can hold a second feature network that can be fed
external features. The model is then trained end-to-end.
"""
class LTR_score(nn.Module):
    def __init__(self, static_feature_size, dropout, hidden_size, feature_model=None):
        super(LTR_score, self).__init__()
        self.feature_model = feature_model
        self.static_feature_size = static_feature_size
        if feature_model is None:
            x_in = static_feature_size
        else:
            x_in = feature_model.feature_size + static_feature_size

        self.hidden = torch.nn.Linear(x_in, hidden_size)   # hidden layer
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.predict = torch.nn.Linear(hidden_size, 1) 

    def forward(self, image, static_features, saliency=None):
        if self.feature_model is not None:
            if saliency is not None:
                image = self.feature_model(image, saliency)
            else:
                image = self.feature_model(image)

            if self.static_feature_size == 0:
                features = image
            else:
                if static_features.dim() == 1:
                    static_features = static_features.unsqueeze(0)

                # Most dirty hack in the history of hacks, but don't want to figure out this problem right now.
                if type(image) is tuple:
                    image = image[0]
                if type(static_features) is tuple:
                    static_features = static_features[0]

                features = torch.cat((image, static_features), 1)
        else:
            features = static_features

        x = self.hidden(features)
        x = self.relu(x)
        total_units = x.size(0) * x.size(1)
        if torch.nonzero(x).size(0) / float(total_units) < 0.20:
            logger.warning("dead relu's, only {0:.2f} of units are alive".format(torch.nonzero(x).size(0) / float(total_units)))
            
        x = self.dropout(x)
        x = self.predict(x)  

        return x

