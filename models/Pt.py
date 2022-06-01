from torch.utils.data import TensorDataset
import torch
from sentence_transformers import SentenceTransformer 
from .OneLayerModel import OneLayerModel

class Pt():
    def __init__(self, device):
        self.device = device
        self.embeder= SentenceTransformer('weights/multi', device=device)
        self.oneLayerModel = OneLayerModel(768, 2940)
        self.oneLayerModel.load_state_dict(torch.load('weights/pt/model.pt'))
        self.oneLayerModel.to(device)

    def classify(self, texts):
        embeds = self.embeder.encode(texts)
        return self.oneLayerModel(torch.from_numpy(embeds).to(self.device))