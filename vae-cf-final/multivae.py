import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


class GateLinear(nn.Module):
    def __init__(self, input_size, output_size, gated=False):
        super(GateLinear, self).__init__()

        self.gated = gated
        self.linear = nn.Linear(int(input_size), int(output_size))
        if self.gated:
            self.sigmoid = nn.Sigmoid()
            self.g = nn.Linear(int(input_size), int(output_size))

    def forward(self, x):
        h = self.linear(x)
        if self.gated:
            g = self.sigmoid(self.g(x))
            h = h * g
        return h


class MultiVAE(nn.Module):
    def __init__(self, data, device, q_dims, p_dims, gated=False, dropout=0):
        super(MultiVAE, self).__init__()
        self.data = data
        self.device = device
        self.q_dims = q_dims
        self.p_dims = p_dims
        self.gated = gated
        self.dropout = dropout
        self.__init_weights()

    def __init_weights(self):

        self.dropout = nn.Dropout(p=self.dropout)
        self.activation = nn.Tanh()

        # encoder layers
        self.q_dims = [self.data.num_items] + self.q_dims
        self.q_dims[-1] = self.q_dims[-1]*2
        self.q_layers = []
        for i in range(1, len(self.q_dims)):
            if i == 1:
                self.q_layers.append(nn.Linear(self.q_dims[i-1], self.q_dims[i], bias=False))
            else:
                self.q_layers.append(GateLinear(self.q_dims[i-1], self.q_dims[i], self.gated))
        self.q_layers = nn.ModuleList(self.q_layers)

        # decoder layers
        self.p_dims = self.p_dims + [self.data.num_items]
        assert self.q_dims[-1]/2 == self.p_dims[0]
        self.p_layers = []
        for i in range(1, len(self.p_dims)-1):
            self.p_layers.append(GateLinear(self.p_dims[i-1], self.p_dims[i], self.gated))
        self.p_layers.append(nn.Linear(self.p_dims[-2], self.p_dims[-1], bias=False))
        self.p_layers = nn.ModuleList(self.p_layers)

        # generate item features
        self.feat_matrix = convert_sp_mat_to_sp_tensor(self.data.item_features).to(self.device)
        self.cooccurence = convert_sp_mat_to_sp_tensor(self.data.cooccurence).to(self.device)

        # month embs
        self.month_embs = torch.nn.Embedding(num_embeddings=self.data.num_month, embedding_dim=self.q_dims[1])

        # feat layers
        self.en_fc_1 = nn.Linear(self.feat_matrix.shape[1], self.q_dims[1])
        self.en_fc_2 = nn.Linear(self.q_dims[1], self.q_dims[1])
        self.de_fc_1 = nn.Linear(self.feat_matrix.shape[1], self.p_dims[-2])
        self.de_fc_2 = nn.Linear(self.p_dims[-2], self.p_dims[-2])

    def get_item_feature(self):
        encoder_item_feature = self.en_fc_2(self.activation(self.en_fc_1(self.feat_matrix)))
        decoder_item_feature = self.de_fc_2(self.activation(self.de_fc_1(self.feat_matrix)))

        return encoder_item_feature, decoder_item_feature

    def encode(self, x, item_feat=None, month_feature=None):
        h = x

        for i, layer in enumerate(self.q_layers):
            if i == 0 and item_feat != None:

                embs = [layer.weight.T]
                embs.append(torch.sparse.mm(self.cooccurence, embs[-1]))
                embs = torch.sum(torch.stack(embs, dim=1), dim=1)

                h = torch.matmul(h, embs+item_feat)
            else:
                h = layer(h)
            if i != len(self.q_layers) - 1:
                # h += month_feature
                h = self.activation(h)

        mu = h[:, :self.p_dims[0]]
        logvar = h[:, self.p_dims[0]:]

        return mu, logvar
    
    def decode(self, z, item_feat=None):
        h = z
        for i, layer in enumerate(self.p_layers):
            if i == len(self.p_layers) - 1 and item_feat != None:

                embs = [layer.weight]
                embs.append(torch.sparse.mm(self.cooccurence, embs[-1]))
                embs = torch.sum(torch.stack(embs, dim=1), dim=1)

                h = torch.matmul(h, (embs+item_feat).T)
            else:
                h = layer(h)

            if i != len(self.p_layers) - 1:
                h = self.activation(h)

        return h

    def computer(self, x, x_m):

        if self.dropout != 0:
            h = self.dropout(x.to_dense())
        else:
            h = x.to_dense()

        month_feature = self.month_embs(x_m)
        encoder_item_feature, decoder_item_feature = self.get_item_feature()
        mu, logvar = self.encode(h, encoder_item_feature, month_feature)
        recon_x = self.decode(mu, decoder_item_feature)

        return recon_x, mu, logvar

    def compute_rating(self):

        candidates = torch.from_numpy(self.data.candidate_val).long().to(self.device)

        N = self.data.X_val.shape[0]
        idxlist = list(range(N))

        ratings = []
        for start_idx in range(0, N, 5000):
            end_idx = min(start_idx+5000, N)

            x = self.data.X_val[idxlist[start_idx:end_idx]]
            x_in = convert_sp_mat_to_sp_tensor(x).to(self.device)
            x = convert_sp_mat_to_sp_tensor(x).to(self.device).to_dense()

            x_m = torch.LongTensor(self.data.X_m_val[start_idx:end_idx]).to(self.device)

            recon_x, _, _ = self.computer(x_in, x_m)

            recon_x[x>0] = -float('inf')
            recon_x = recon_x[:, candidates]
            ratings.append(recon_x.detach().cpu())

        return torch.cat(ratings, 0)
