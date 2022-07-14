import argparse
import gc
from collections import OrderedDict
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import NUM_FEATURE_CATEGORY_ID, VAR_LEN_FEATURE_CATEGORY_ID_LIST, MONTH_LIST
from file_utils import read_pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, choices=['leaderboard', 'final'], default='leaderboard')
    parser.add_argument('--batch_size', type=int, default=512)
    return parser.parse_args()


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.item_encoder = ItemEncoder(args=args)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=args.transformer_d_model,
                nhead=args.transformer_nhead,
                dim_feedforward=args.transformer_dim_feedforward,
                dropout=args.transformer_dropout,
                activation=args.transformer_activation
            ),
            num_layers=args.transformer_num_layers,
            norm=nn.LayerNorm(normalized_shape=args.transformer_d_model)
        )
        self.position_embedding = nn.Embedding(num_embeddings=args.context_max_len + 1, embedding_dim=args.transformer_d_model)
        self.month_embedding = nn.Embedding(num_embeddings=len(MONTH_LIST), embedding_dim=args.transformer_d_model)

    def encode_context(self, item_encoded: torch.Tensor, context_item_indices_batch: torch.Tensor, month_index_batch: torch.Tensor):
        context_item_encoded = item_encoded[context_item_indices_batch, :]
        position_encoded = self.position_embedding(torch.arange(context_item_encoded.size(1), device='cuda')).unsqueeze(0)
        month_encoded = self.month_embedding(month_index_batch).unsqueeze(1)
        context_item_encoded = context_item_encoded + position_encoded + month_encoded
        context_encoded = self.transformer(src=context_item_encoded.transpose(0, 1), src_key_padding_mask=context_item_indices_batch == 0).transpose(0, 1)
        return context_encoded


class ItemEncoder(nn.Module):
    def __init__(self, args):
        super(ItemEncoder, self).__init__()
        self.args = args
        self.item_id_embedding = nn.Embedding(num_embeddings=args.num_item_ids, embedding_dim=args.item_embedding_dim)
        self.item_fixed_len_features_embedding = nn.Embedding(num_embeddings=args.num_item_fixed_len_features, embedding_dim=args.feature_embedding_dim)
        self.item_var_len_features_embedding = nn.ModuleList([nn.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=args.feature_embedding_dim) for num_embeddings in [16, 6, 67, 4, 5]])
        in_features = args.item_embedding_dim + args.feature_embedding_dim * (args.num_item_fixed_len_feature_category_ids + 5)
        self.fc = nn.Linear(in_features=in_features, out_features=args.transformer_d_model)
        self.pad_token_embedded = nn.Parameter(torch.zeros(1, args.transformer_d_model), requires_grad=False)
        self.mask_token_embedded = nn.Parameter(torch.zeros(1, args.transformer_d_model), requires_grad=True)

    def forward(self, item_id_batch: torch.Tensor, item_fixed_len_features_batch: torch.Tensor, item_var_len_features_batch: List[torch.Tensor], item_var_len_features_offsets_batch: torch.Tensor):
        item_id_embedded = self.item_id_embedding(item_id_batch)
        item_fixed_len_features_embedded = self.item_fixed_len_features_embedding(item_fixed_len_features_batch).flatten(1, 2)
        item_var_len_features_embedded = torch.cat([self.item_var_len_features_embedding[i](item_var_len_features_batch[i], item_var_len_features_offsets_batch[i, :]) for i in range(5)], dim=1)
        item_encoded = self.fc(torch.cat([item_id_embedded, item_fixed_len_features_embedded, item_var_len_features_embedded], dim=1))
        item_encoded = torch.cat([self.pad_token_embedded, self.mask_token_embedded, item_encoded], dim=0)
        return item_encoded


class TestDataset(Dataset):
    def __init__(self, args, test_sessions, test_item_index_dict):
        self.args = args
        self.test_sessions = test_sessions
        self.test_item_index_dict = test_item_index_dict

    def __getitem__(self, index):
        context = self.test_sessions[index]
        context = context[-self.args.context_max_len:][::-1]
        context_item_ids, context_times = zip(*context)
        context_item_indices = [1] + [self.test_item_index_dict[item_id] for item_id in context_item_ids]
        month_index = len(MONTH_LIST) - 1
        return context_item_indices, month_index

    def __len__(self):
        return len(self.test_sessions)


def test_collate_fn(batch):
    context_item_indices_list, month_index_list = zip(*batch)
    context_max_len = max(len(context_item_indices) for context_item_indices in context_item_indices_list)
    context_item_indices_batch = torch.LongTensor([context_item_indices + [0] * (context_max_len - len(context_item_indices)) for context_item_indices in context_item_indices_list])
    month_index_batch = torch.LongTensor(month_index_list)
    return context_item_indices_batch, month_index_batch


def get_item_fixed_len_feature_category_ids():
    return [
        feature_category_id for feature_category_id in range(1, NUM_FEATURE_CATEGORY_ID + 1)
        if feature_category_id != 27 and feature_category_id not in VAR_LEN_FEATURE_CATEGORY_ID_LIST
    ]


def merge_item_features(item_features):
    item_features = deepcopy(item_features)
    item_fixed_len_feature_category_ids = get_item_fixed_len_feature_category_ids()
    for feature_category_id in item_fixed_len_feature_category_ids:
        for item_id in item_features:
            if feature_category_id in item_features[item_id]:
                if len(item_features[item_id][feature_category_id]) == 1:
                    item_features[item_id][feature_category_id] = item_features[item_id][feature_category_id][0]
                else:
                    assert feature_category_id == 1 and sorted(item_features[item_id][feature_category_id]) == [461, 771]
                    item_features[item_id][feature_category_id] = 461
    return item_features


def preprocess_item_features(item_index_dict, item_id_dict, item_features, feature_dict):
    item_fixed_len_feature_category_ids = get_item_fixed_len_feature_category_ids()
    item_fixed_len_features_offsets = [0] + np.cumsum([
        len(feature_dict[feature_category_id]) for feature_category_id in item_fixed_len_feature_category_ids
    ]).tolist()[: -1]
    item_id_list, item_fixed_len_features_list = [], []
    item_var_len_features_list = [[] for _ in range(len(VAR_LEN_FEATURE_CATEGORY_ID_LIST))]
    item_var_len_features_offsets_list = [[] for _ in range(len(VAR_LEN_FEATURE_CATEGORY_ID_LIST))]
    var_len_features_offsets = [0] * len(VAR_LEN_FEATURE_CATEGORY_ID_LIST)
    for item_id in item_index_dict:
        if item_id == '[PAD]' or item_id == '[MASK]':
            continue
        item_id_list.append(item_id_dict.get(item_id, 1))
        item_fixed_len_features = []
        for feature_category_id, feature_offset in zip(item_fixed_len_feature_category_ids, item_fixed_len_features_offsets):
            if feature_category_id in item_features[item_id]:
                feature_index = feature_dict[feature_category_id][item_features[item_id][feature_category_id]]
            else:
                feature_index = 0
            feature_index += feature_offset
            item_fixed_len_features.append(feature_index)
        item_fixed_len_features_list.append(item_fixed_len_features)
        for i, feature_category_id in enumerate(VAR_LEN_FEATURE_CATEGORY_ID_LIST):
            item_var_len_features_offsets_list[i].append(var_len_features_offsets[i])
            if feature_category_id in item_features[item_id]:
                for feature_value_id in item_features[item_id][feature_category_id]:
                    item_var_len_features_list[i].append(feature_dict[feature_category_id][feature_value_id])
                var_len_features_offsets[i] += len(item_features[item_id][feature_category_id])
    return item_id_list, item_fixed_len_features_list, item_var_len_features_list, item_var_len_features_offsets_list


def build_item_index_dict(item_counts):
    item_index_dict = OrderedDict([('[PAD]', 0), ('[MASK]', 1)])
    for item_id in item_counts:
        item_index_dict[item_id] = len(item_index_dict)
    return item_index_dict


def main():
    args = parse_args()
    item_features = pd.read_csv('data/item_features.csv')
    item_ids = sorted(item_features['item_id'].unique())
    item_id_dict = {'[UNK]': 0, '[MASK]': 1}
    for item_id in item_ids:
        item_id_dict[item_id] = len(item_id_dict)
    feature_dict = read_pickle('cache/feature_dict.pkl')
    test_session_ids = read_pickle(f'cache/test_{args.split}_session_ids.pkl')
    test_sessions = read_pickle(path=f'cache/test_{args.split}_sessions.pkl')
    test_candidate_items = read_pickle(path=f'cache/test_candidate_items.pkl')
    item_features = read_pickle(path=f'cache/item_features.pkl')
    item_features = merge_item_features(item_features=item_features)
    train_valid_item_counts = read_pickle(path=f'cache/train_valid_item_counts.pkl')
    for session in test_sessions:
        for item_id, _ in session:
            if item_id not in train_valid_item_counts:
                train_valid_item_counts[item_id] = 0
    for item_id in test_candidate_items:
        if item_id not in train_valid_item_counts:
            train_valid_item_counts[item_id] = 0
    test_item_index_dict = build_item_index_dict(item_counts=train_valid_item_counts)
    (
        test_item_id_list,
        test_item_fixed_len_features_list,
        test_item_var_len_features_list,
        test_item_var_len_features_offsets_list
    ) = preprocess_item_features(
        item_index_dict=test_item_index_dict,
        item_id_dict=item_id_dict,
        item_features=item_features,
        feature_dict=feature_dict
    )
    test_item_id_batch = torch.LongTensor(test_item_id_list).cuda()
    test_item_fixed_len_features_batch = torch.LongTensor(test_item_fixed_len_features_list).cuda()
    test_item_var_len_features_batch = [torch.LongTensor(item).cuda() for item in test_item_var_len_features_list]
    test_item_var_len_features_offsets_batch = torch.LongTensor(test_item_var_len_features_offsets_list).cuda()
    test_candidate_item_indices_batch = [test_item_index_dict[item_id] for item_id in test_candidate_items]
    test_candidate_item_indices_batch = torch.LongTensor(test_candidate_item_indices_batch).cuda()
    test_context_indices_list = [sorted(set(item_id for item_id, _ in session)) for session in test_sessions]
    test_context_indices_list = [
        [test_candidate_items.index(item_id) for item_id in session if item_id in test_candidate_items]
        for session in test_context_indices_list
    ]
    test_predictions_mask = np.zeros(shape=(len(test_sessions), len(test_candidate_items)), dtype=np.float32)
    for i, context_indices in enumerate(test_context_indices_list):
        test_predictions_mask[i, context_indices] = -1000000.0
    final_prediction_scores = np.zeros(shape=(len(test_sessions), len(test_candidate_items)), dtype=np.float32)
    normalization_term = 0.0
    for model_index in range(1, 60 + 1):
        model_args = read_pickle(f'model/dot_product_v{model_index}_args.pkl')
        model_args.num_item_ids = len(item_id_dict)
        model_args.item_fixed_len_feature_category_ids = get_item_fixed_len_feature_category_ids()
        model_args.num_item_fixed_len_feature_category_ids = len(model_args.item_fixed_len_feature_category_ids)
        model_args.num_item_fixed_len_features = sum(
            len(feature_dict[feature_category_id]) for feature_category_id in model_args.item_fixed_len_feature_category_ids
        )

        test_dataset = TestDataset(
            args=model_args,
            test_sessions=test_sessions,
            test_item_index_dict=test_item_index_dict
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=test_collate_fn,
            pin_memory=True,
            drop_last=False
        )
        model = Model(args=model_args)
        model.load_state_dict(torch.load(f'model/dot_product_v{model_index}.bin'))
        model.cuda()
        model.eval()
        with torch.no_grad():
            test_context_encoded = []
            item_encoded = model.item_encoder(
                item_id_batch=test_item_id_batch,
                item_fixed_len_features_batch=test_item_fixed_len_features_batch,
                item_var_len_features_batch=test_item_var_len_features_batch,
                item_var_len_features_offsets_batch=test_item_var_len_features_offsets_batch
            )
            for context_item_indices_batch, month_index_batch in tqdm(test_dataloader):
                context_item_indices_batch = context_item_indices_batch.cuda()
                month_index_batch = month_index_batch.cuda()
                context_encoded = model.encode_context(
                    item_encoded=item_encoded,
                    context_item_indices_batch=context_item_indices_batch,
                    month_index_batch=month_index_batch
                )[:, 0, :]
                test_context_encoded.append(context_encoded.cpu().numpy())
            test_item_encoded = item_encoded[test_candidate_item_indices_batch, :]
            test_context_encoded = np.concatenate(test_context_encoded, axis=0)
            test_context_encoded = torch.from_numpy(test_context_encoded).cuda()
            test_predictions = torch.matmul(test_context_encoded, test_item_encoded.transpose(0, 1))
            test_predictions += torch.from_numpy(test_predictions_mask).cuda()
            test_prediction_scores = test_predictions.cpu().numpy()
            test_prediction_scores[test_predictions_mask == -1000000.0] = float('-inf')
            if model_index in [7, 8, 11, 15, 17, 27, 32, 34, 40, 46, 54, 59]:
                final_prediction_scores += test_prediction_scores * 0.95
                normalization_term += 0.95
            elif model_index in [1, 12, 21, 26, 30, 36, 37, 42, 45, 47, 52, 53]:
                final_prediction_scores += test_prediction_scores * 0.99
                normalization_term += 0.99
            else:
                final_prediction_scores += test_prediction_scores * 1.0
                normalization_term += 1.0
        del model, test_dataset, test_dataloader
        gc.collect()
        torch.cuda.empty_cache()
    final_prediction_scores /= normalization_term
    with open(f'output/{args.split}_raw_scores.csv', 'w') as file_out:
        for i, session_id in enumerate(tqdm(test_session_ids)):
            for j, item_id in enumerate(test_candidate_items):
                file_out.write(f'{session_id},{item_id},{final_prediction_scores[i][j]}\n')


if __name__ == '__main__':
    main()
