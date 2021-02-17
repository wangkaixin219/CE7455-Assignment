import torch
import torch.nn as nn


class FNNModel(nn.Module):

    def __init__(self, n_tokens, in_dim, hid_dim, tie_weights=True):
        super(FNNModel, self).__init__()
        self.n_tokens = n_tokens
        self.encoder = nn.Embedding(n_tokens, in_dim)
        self.layer = nn.Linear(in_dim, hid_dim)
        self.decoder = nn.Linear(hid_dim, n_tokens)

        if tie_weights:
            if hid_dim != in_dim:
                raise ValueError('When using the tied flag, hid_dim must be equal to in_dim')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.encoder.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)

    def forward(self, x):
        emb = self.encoder(x)
        output = self.layer(emb)
        output = torch.tanh(output)
        decoded = self.decoder(output)
        return decoded 
