from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models, losses
import torch

class EnV2(torch.nn.Module):

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def __init__(self):
        super(EnV2, self).__init__()

        self.embed_dim = 768
        self.num_class = 2942
        base_name = 'all-mpnet-base-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(f'sentence-transformers/{base_name}')
        self.sbert_model = AutoModel.from_pretrained(f'sentence-transformers/{base_name}')
        self.fc1 = torch.nn.Linear(self.embed_dim, self.num_class)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.5
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()

    def forward(self, encodes):
        model_output = self.sbert_model(**encodes)
        sentence_embeddings = self.mean_pooling(model_output, encodes['attention_mask'])
        x = self.fc1(sentence_embeddings)
        return x
    


