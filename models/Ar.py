from sentence_transformers import SentenceTransformer
import torch
import pickle

class Ar():

    def load_train_embeds(self, device):
        embeds_path  = './weights/ar/embeds.pickle'
        model_path = './weights/ar/model_data' 
        with open(embeds_path,'rb') as f:
            train_embeds = pickle.load(f)
        train_embeds = torch.from_numpy(train_embeds).to(self.device)
        model = SentenceTransformer(model_path, device=device)
        return train_embeds, model

    def __init__(self, device):
        super(Ar, self).__init__()
        self.device = device
        self.train_embeds, self.model = self.load_train_embeds(device)
        

    def cosine_similarity(self, embeds):
        return torch.mm(embeds, self.train_embeds.T)
    
    def classify(self, texts):
        embeds = self.model.encode(texts)
        return self.cosine_similarity(torch.from_numpy(embeds).to(self.device))