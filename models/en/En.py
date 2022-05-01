from sentence_transformers import SentenceTransformer
import torch
import pickle

class En():

    def load_train_embeds(self, ):
        embeds_path  = './models/en/embeds.pickle'
        with open(embeds_path,'rb') as f:
            train_embeds = pickle.load(f)
        return train_embeds

    def __init__(self, device):
        super(En, self).__init__()
        base_name = "all-mpnet-base-v2"
        self.device = device
        self.train_embeds = self.load_train_embeds()
        self.train_embeds = self.train_embeds.to(device)
        self.sbert_model = SentenceTransformer(base_name, device=device)

    def cosine_similarity(self, embeds):
        return torch.mm(embeds, self.train_embeds.T)
    
    def classify(self, texts):
        embeds = self.sbert_model.encode(texts)
        return self.cosine_similarity(torch.from_numpy(embeds).to(self.device))