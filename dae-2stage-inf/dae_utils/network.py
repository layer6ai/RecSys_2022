# https://github.com/ryancheunggit/tabular_dae/tree/main

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import joblib
import math


def _make_mlp_layers(num_units):
    layers = nn.ModuleList([
        nn.Linear(in_features, out_features)
        for in_features, out_features in zip(num_units, num_units[1:])
    ])
    return layers


class TransformerEncoder(nn.Module):
    ''' Transformer Encoder. '''

    def __init__(self, embed_dim, num_heads, dropout, feedforward_dim):
        super().__init__()
        assert embed_dim % num_heads == 0, 'embed_dim must be a multiple of num_heads'
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear_1 = nn.Linear(embed_dim, feedforward_dim)
        self.linear_2 = nn.Linear(feedforward_dim, embed_dim)
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x_in):
        ''' input is of shape num_subspaces x batch_size x embed_dim '''
        attn_out, _ = self.attn(x_in, x_in, x_in)
        x = self.layernorm_1(x_in + attn_out)
        ff_out = self.linear_2(F.relu(self.linear_1(x)))
        x = self.layernorm_2(x + ff_out)
        return x


class Transformer(nn.Module):
    ''' DAE Body with transformer encoders. '''

    def __init__(self, in_features, hidden_size=1024, num_subspaces=8, embed_dim=128, num_heads=8, dropout=0,
                 feedforward_dim=512, num_layers=3):
        super().__init__()
        assert hidden_size == embed_dim * num_subspaces, 'num_subspaces must be a multiple of embed_dim'
        self.num_subspaces = num_subspaces
        self.embed_dim = embed_dim
        self.excite = nn.Linear(in_features, hidden_size)
        self.encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim)
            for _ in range(num_layers)
        ])
        self._output_shape = hidden_size

    @property
    def output_shape(self): return self._output_shape

    def divide(self, x):
        return rearrange(x, 'b (s e) -> s b e', s=self.num_subspaces, e=self.embed_dim)

    def combine(self, x):
        return rearrange(x, 's b e -> b (s e)')

    def forward_pass(self, x):
        outputs = []
        x = F.relu(self.excite(x))
        outputs.append(x)
        x = self.divide(x)
        for encoder in self.encoders:
            x = encoder(x)
            outputs.append(x)
        return outputs

    def forward(self, x):
        return self.combine(self.forward_pass(x)[~0])

    def featurize(self, x):
        return torch.cat([self.combine(x) if i == 1 else x for i, x in enumerate(self.forward_pass(x))], dim=1)


class ReconNet(nn.Module):
    def __init__(self, in_features, n_nums=0, dropout=0.2):
        super().__init__()
        self.n_nums = n_nums
        self.nums_linear = nn.Linear(in_features, n_nums) if n_nums else None
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, features):
        outputs = dict()
        outputs['nums'] = self.nums_linear(self.dropout_layer(features))
        return outputs


class AutoEncoder(nn.Module):
    def __init__(self, datatype_info, cutoff=0, cross=0, loss_type='sampled_ce', temp=1, item_content=None,
                 item_content_drop=0., decode_feature=0, item_knn=None,
                 item_embed_dim=None, body_network=Transformer, body_network_cfg=dict()):
        super().__init__()
        self.n_nums = datatype_info.get('n_nums', 0)
        self.body_network = body_network
        if isinstance(body_network, str):
            body_network = _ae_body_options[body_network]

        self.temp = temp
        self.cutoff = cutoff

        self.body_network_in_features = self.n_nums
        self.body_network_cfg = body_network_cfg
        self.item_content = item_content
        self.item_knn = item_knn
        self.datatype_info = datatype_info
        self.item_embed_dim = item_embed_dim
        self.feature_drop_layer = nn.Dropout(p=item_content_drop)
        self.decode_feature = decode_feature
        self.cross = cross

        in_dim = self.body_network_in_features
        feature_dim = 0
        if item_embed_dim is not None:
            self.item_embed = torch.nn.Embedding(num_embeddings=item_embed_dim.shape[0],
                                                 embedding_dim=item_embed_dim.shape[1])
            self.item_embed.weight = torch.nn.Parameter(torch.FloatTensor(item_embed_dim))
            # self.item_embed.weight.requires_grad = False
            in_dim += item_embed_dim.shape[1]
            feature_dim += item_embed_dim.shape[1]

        if item_content is not None:
            self.item_content_embed = torch.nn.Embedding(num_embeddings=item_content.shape[0],
                                                         embedding_dim=item_content.shape[1])
            self.item_content_embed.weight = torch.nn.Parameter(torch.FloatTensor(item_content))
            self.item_content_embed.weight.requires_grad = False
            in_dim += item_content.shape[1]
            feature_dim += item_content.shape[1]

        if item_knn is not None:
            self.itemknn_embed = torch.nn.Embedding(num_embeddings=item_knn.shape[0],
                                                    embedding_dim=item_knn.shape[1])
            self.itemknn_embed.weight = torch.nn.Parameter(torch.FloatTensor(item_knn))
            self.itemknn_embed.weight.requires_grad = False
            in_dim += item_knn.shape[1]
            feature_dim += item_knn.shape[1]

        if self.decode_feature > 1:
            self.item_content_lookup = torch.nn.Embedding(num_embeddings=item_content.shape[1],
                                                          embedding_dim=self.decode_feature)
            in_dim += self.decode_feature

        self.body = body_network(in_features=in_dim, **body_network_cfg)

        if self.cross:
            self.input_cross = nn.Linear(in_dim, 1)
            torch.nn.init.normal_(self.input_cross.weight, 0, 1. / math.sqrt(self.input_cross.weight.size()[1]))
            self.output_cross = nn.Linear(self.body.output_shape, 1)
            torch.nn.init.normal_(self.output_cross.weight, 0, 1. / math.sqrt(self.output_cross.weight.size()[1]))

        if self.decode_feature == 1:
            self.feature_fuse = nn.Linear(item_content.shape[1], self.body.output_shape)
            self.feature_fuse_dropout = nn.Dropout(p=item_content_drop)
            self.feature_fuse2 = nn.Linear(self.feature_fuse.out_features + self.body.output_shape,
                                           self.body.output_shape)
            self.reconstruction_head = ReconNet(self.body.output_shape, item_content.shape[0], dropout=0)

        else:
            self.reconstruction_head = ReconNet(self.body.output_shape, item_content.shape[0],
                                                dropout=item_content_drop)
        self.mask_predictor_head = nn.Linear(self.body.output_shape, item_content.shape[0])

    def process_inputs(self, inputs):
        x_nums = inputs.get('nums', None)
        body_network_inputs = torch.cat([t for t in [x_nums] if t is not None], dim=1)
        return body_network_inputs

    def forward(self, inputs, featurize=False):
        if len(inputs) > 1:
            body_network_inputs = self.process_inputs(inputs[0])
        else:
            body_network_inputs = self.process_inputs(inputs)

        if self.cutoff:
            kvals, kidx = body_network_inputs.topk(self.cutoff, dim=1)
            weighted_item_content = torch.einsum('bi,bij->bj', kvals, self.item_content_embed(kidx))
        else:
            weighted_item_content = body_network_inputs @ self.item_content_embed.weight

        feature = torch.cat((body_network_inputs, weighted_item_content), dim=1)
        if len(inputs) > 1:
            feature = torch.cat((feature, inputs[1]), dim=1)

        if self.item_knn is not None:
            weighted_item_knn = body_network_inputs @ self.itemknn_embed.weight
            feature = torch.cat((feature, weighted_item_knn), dim=1)
        if self.item_embed_dim is not None:
            weighted_item_embed = body_network_inputs @ self.item_embed.weight
            feature = torch.cat((feature, weighted_item_embed), dim=1)
        if self.decode_feature > 1:
            weighted_item_lookup = body_network_inputs @ (self.item_content_embed.weight != 0).float()
            weighted_item_lookup = weighted_item_lookup @ self.item_content_lookup.weight
            feature = torch.cat((feature, weighted_item_lookup), dim=1)
        if self.cross:
            feature = (self.input_cross.weight + 1) * feature

        last_hidden = self.body(feature)

        if self.decode_feature == 1:
            fused = self.feature_fuse(self.feature_fuse_dropout(weighted_item_content))  # 977->128
            reconstruction = self.reconstruction_head(self.feature_fuse2(torch.cat((last_hidden, fused), dim=1)))
        else:
            reconstruction = self.reconstruction_head(last_hidden)
        predicted_mask = self.mask_predictor_head(last_hidden)
        if not featurize:
            return reconstruction, predicted_mask
        else:
            featurized = self.body.featurize(feature)
            return reconstruction, featurized

    def featurize(self, inputs):
        body_network_inputs = self.process_inputs(inputs)
        weighted_item_content = body_network_inputs @ self.feature_drop_layer(self.item_content_embed.weight)
        feature = torch.cat((body_network_inputs, weighted_item_content), dim=1)
        features = self.body.featurize(feature)
        return features

    def inference(self, max_k, data, device, batch_size):
        candidates = torch.from_numpy(data.candidate_val).long().to(device)
        N = data.X_val.shape[0]
        idxlist = list(range(N))
        rating_list = []
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            x = data.X_val[idxlist[start_idx:end_idx]]
            x = torch.FloatTensor(x).to(device)
            if data.args.sess_feature:
                sess = data.val_sess_feature[idxlist[start_idx:end_idx]]
                sess = torch.FloatTensor(sess).to(device)
                # x = torch.cat((x, sess), dim=1)
                logit, _ = self.forward(({'nums': x}, sess))
            else:
                logit, _ = self.forward({'nums': x})
            logit = logit['nums']
            logit[x > 0] = -(1 << 10)
            logit_candidate = logit[:, candidates]
            _, rating_K = torch.topk(logit_candidate, k=max_k)
            rating_list.append(rating_K.cpu().numpy())
        rating_list = np.concatenate(rating_list)
        rating_list = data.candidate_val[rating_list]
        return rating_list

    def save(self, path_to_model_dump, args, result=None):
        model_state_dict = dict(
            constructor_args=dict(
                datatype_info=self.datatype_info,
                item_content=self.item_content,
                body_network=self.body_network,
                body_network_cfg=self.body_network_cfg,
            ),
            result=result,
            args=args,
            network_state_dict={k: v.cpu() for k, v in self.state_dict().items()}
        )
        joblib.dump(model_state_dict, path_to_model_dump)


_ae_body_options = {
    'transformer': Transformer
}
