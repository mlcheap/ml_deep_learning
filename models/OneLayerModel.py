import torch

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
