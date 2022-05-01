from torch.utils.data import TensorDataset
import torch
from sentence_transformers import SentenceTransformer 

class OneLayerModel(torch.nn.Module):

    def __init__(self, embed_dim, num_class):
        super(OneLayerModel, self).__init__()
        self.fc1 = torch.nn.Linear(embed_dim, num_class)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.5
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()

    def forward(self, embedded):
        x = self.fc1(embedded)
        return x


class Pt():
    def __init__(self, device):
        self.device = device
        self.embeder= SentenceTransformer('models/multi', device=device)
        self.oneLayerModel = OneLayerModel(768, 3561)
        self.oneLayerModel.load_state_dict(torch.load('models/pt/model.pt'))
        self.oneLayerModel.to(device)

    def classify(self, texts):
        embeds = self.embeder.encode(texts)
        return self.oneLayerModel(torch.from_numpy(embeds).to(self.device))