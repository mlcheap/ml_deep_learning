from sentence_transformers import SentenceTransformer
import torch
import pickle
import io 

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
            return lambda b: torch.load(io.BytesIO(b), map_location=DEVICE)
        else:
            return super().find_class(module, name)

class En():

    def load_train_embeds(self, ):
        embeds_path  = './models/weights/en/embeds.pickle'
        with open(embeds_path,'rb') as f:
            train_embeds = CPU_Unpickler(f).load()
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