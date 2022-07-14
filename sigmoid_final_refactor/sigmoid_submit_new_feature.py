import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from typing import List

from config import DATA_DIR, CACHE_DIR, MODEL_DIR, SUBMISSION_DIR, MONTH_LIST
from data_utils_sigmoid_train_valid import extract_featuresall, valid_collate_fn
from file_utils import read_pickle
from time_utils import Timer
from train_utils import (
    build_item_index_dict,
    merge_item_features,
    preprocess_item_features,
    get_item_fixed_len_feature_category_ids
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='sigmoid_v1')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=8)
    return parser.parse_args()


class ValidDataset(Dataset):
    def __init__(
            self,
            args,
            valid_sessions,
            valid_candidate_items,
            valid_item_index_dict,
            metadata,
            metadata2m,
            metadataall,
            metadatacurm,
            cooccurrence
    ):
        self.args = args
        self.valid_sessions = valid_sessions
        self.valid_candidate_items = valid_candidate_items
        self.valid_item_index_dict = valid_item_index_dict
        self.metadata = metadata
        self.metadata2m = metadata2m
        self.metadataall = metadataall
        self.metadatacurm = metadatacurm
        self.cooccurrence = cooccurrence

    def __getitem__(self, index):
        context = self.valid_sessions[index]
        context = context[-self.args.context_max_len:][::-1]
        context_item_ids, context_times = zip(*context)
        context_item_indices = [self.valid_item_index_dict[item_id] for item_id in context_item_ids]
        month_index = len(MONTH_LIST) - 1
        dense_features = extract_featuresall(
            context_item_ids=context_item_ids,
            candidate_item_ids=self.valid_candidate_items,
            metadata=self.metadata,
            metadata2m=self.metadata2m,
            metadataall=self.metadataall,
            metadatacurm=self.metadatacurm,
            cooccurrence=self.cooccurrence
        )
        return (
            context_item_indices,
            month_index,
            dense_features
        )

    def __len__(self):
        return len(self.valid_sessions)


class ItemEncoder(nn.Module):
    def __init__(self, args):
        super(ItemEncoder, self).__init__()
        self.args = args
        self.item_id_embedding = nn.Embedding(
            num_embeddings=args.num_item_ids,
            embedding_dim=args.item_embedding_dim
        )
        self.item_fixed_len_features_embedding = nn.Embedding(
            num_embeddings=args.num_item_fixed_len_features,
            embedding_dim=args.feature_embedding_dim
        )
        self.item_var_len_features_embedding = nn.ModuleList([
            nn.EmbeddingBag(
                num_embeddings=num_embeddings,
                embedding_dim=args.feature_embedding_dim
            ) for num_embeddings in [16, 6, 67, 4, 5]])
        self.fc = nn.Linear(
            in_features=args.item_embedding_dim + args.feature_embedding_dim * (
                    args.num_item_fixed_len_feature_category_ids + 5),
            out_features=args.transformer_d_model
        )
        self.pad_token_embedded = nn.Parameter(torch.zeros(1, args.transformer_d_model), requires_grad=False)
        self.mask_token_embedded = nn.Parameter(torch.zeros(1, args.transformer_d_model), requires_grad=True)

    def forward(
            self,
            item_id_batch: torch.Tensor,
            item_fixed_len_features_batch: torch.Tensor,
            item_var_len_features_batch: List[torch.Tensor],
            item_var_len_features_offsets_batch: torch.Tensor
    ):
        item_id_embedded = self.item_id_embedding(item_id_batch)
        item_fixed_len_features_embedded = self.item_fixed_len_features_embedding(
            item_fixed_len_features_batch).flatten(1, 2)
        item_var_len_features_embedded = torch.cat([
            self.item_var_len_features_embedding[i](
                item_var_len_features_batch[i],
                item_var_len_features_offsets_batch[i, :]
            ) for i in range(5)
        ], dim=1)
        item_encoded = self.fc(torch.cat([
            item_id_embedded,
            item_fixed_len_features_embedded,
            item_var_len_features_embedded
        ], dim=1))
        item_encoded = torch.cat([
            self.pad_token_embedded,
            self.mask_token_embedded,
            item_encoded
        ], dim=0)
        return item_encoded


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
        self.position_embedding = nn.Embedding(
            num_embeddings=args.context_max_len + 1,
            embedding_dim=args.transformer_d_model
        )
        self.month_embedding = nn.Embedding(
            num_embeddings=17,
            embedding_dim=args.transformer_d_model
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_features=args.transformer_d_model + 21),
            # nn.LayerNorm(args.transformer_d_model + 15),
            nn.Dropout(0.1),
            nn.Linear(in_features=args.transformer_d_model + 21, out_features=args.transformer_d_model // 2),
            nn.BatchNorm1d(num_features=args.transformer_d_model // 2),
            # nn.LayerNorm(args.transformer_d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=args.transformer_d_model // 2, out_features=1)
        )

    def encode_context(
            self,
            item_encoded: torch.Tensor,
            context_item_indices_batch: torch.Tensor,
            month_index_batch: torch.Tensor
    ):
        context_item_encoded = item_encoded[context_item_indices_batch, :]
        position_encoded = self.position_embedding(torch.arange(context_item_encoded.size(1), device='cuda')).unsqueeze(
            0)
        month_encoded = self.month_embedding(month_index_batch).unsqueeze(1)
        context_item_encoded = context_item_encoded + position_encoded + month_encoded
        context_encoded = self.transformer(
            src=context_item_encoded.transpose(0, 1),
            src_key_padding_mask=context_item_indices_batch == 0
        ).transpose(0, 1)
        return context_encoded

    def predict(
            self,
            item_encoded: torch.Tensor,
            item_indices_batch: torch.Tensor,
            month_index_batch: torch.Tensor,
            dense_features_batch: torch.Tensor
    ):
        context_item_encoded = item_encoded[item_indices_batch, :]
        position_encoded = self.position_embedding(torch.arange(context_item_encoded.size(1), device='cuda')).unsqueeze(
            0)
        month_encoded = self.month_embedding(month_index_batch).unsqueeze(1)
        context_item_encoded = context_item_encoded + position_encoded + month_encoded
        prediction_batch = self.transformer(
            src=context_item_encoded.transpose(0, 1),
            src_key_padding_mask=item_indices_batch == 0
        )[0, :, :]
        prediction_batch = self.fc(torch.cat([prediction_batch, dense_features_batch.flatten(0, 1)], dim=1)).squeeze(1)
        return prediction_batch


def main():
    submission_args = parse_args()

    item_features = pd.read_csv(f'{DATA_DIR}/item_features.csv')
    item_ids = sorted(item_features['item_id'].unique())
    item_id_dict = {'[UNK]': 0, '[MASK]': 1}
    for item_id in item_ids:
        item_id_dict[item_id] = len(item_id_dict)
    feature_dict = read_pickle(f'{MODEL_DIR}/feature_dict.pkl')

    args = read_pickle(f'{MODEL_DIR}/{submission_args.model_name}_args.pkl')
    args.num_item_ids = len(item_id_dict)
    args.item_fixed_len_feature_category_ids = get_item_fixed_len_feature_category_ids()
    args.num_item_fixed_len_feature_category_ids = len(args.item_fixed_len_feature_category_ids)
    args.num_item_fixed_len_features = sum(
        len(feature_dict[feature_category_id]) for feature_category_id in args.item_fixed_len_feature_category_ids
    )

    # manual overwrite
    args.num_workers = 16

    with Timer(name='read_data'):
        valid_sessions = read_pickle(path=f'{CACHE_DIR}/test_leaderboard_sessions.pkl')
        valid_candidate_items = read_pickle(path=f'{CACHE_DIR}/test_candidate_items.pkl')
        item_features = read_pickle(path=f'{CACHE_DIR}/item_features.pkl')
        train_valid_item_counts = read_pickle(path=f'{CACHE_DIR}/train_valid_item_counts.pkl')
        for session in valid_sessions:
            for item_id, _ in session:
                if item_id not in train_valid_item_counts:
                    train_valid_item_counts[item_id] = 0
        for item_id in valid_candidate_items:
            if item_id not in train_valid_item_counts:
                train_valid_item_counts[item_id] = 0
        metadata = read_pickle(path=f'{CACHE_DIR}/metadata.pkl')
        metadata_all = read_pickle(path=f'{CACHE_DIR}/metadata_all.pkl')
        metadata_2m = read_pickle(path=f'{CACHE_DIR}/metadata_2m.pkl')
        metadata_curm = read_pickle(path=f'{CACHE_DIR}/metadata_curm.pkl')
        cooccurrence = read_pickle(path=f'{CACHE_DIR}/cooccurrence.pkl')
        for key in cooccurrence[len(MONTH_LIST)]:
            cooccurrence[len(MONTH_LIST)][key] = cooccurrence[len(MONTH_LIST)][key].todense()

    with Timer(name='preprocess_data'):
        item_features = merge_item_features(item_features=item_features)
        valid_item_index_dict = build_item_index_dict(item_counts=train_valid_item_counts)
        (
            valid_item_id_list,
            valid_item_fixed_len_features_list,
            valid_item_var_len_features_list,
            valid_item_var_len_features_offsets_list
        ) = preprocess_item_features(
            item_index_dict=valid_item_index_dict,
            item_id_dict=item_id_dict,
            item_features=item_features,
            feature_dict=feature_dict
        )
        valid_item_id_batch = torch.LongTensor(valid_item_id_list).cuda()
        valid_item_fixed_len_features_batch = torch.LongTensor(valid_item_fixed_len_features_list).cuda()
        valid_item_var_len_features_batch = [torch.LongTensor(item).cuda() for item in valid_item_var_len_features_list]
        valid_item_var_len_features_offsets_batch = torch.LongTensor(valid_item_var_len_features_offsets_list).cuda()
        valid_context_indices_list = [sorted(set(item_id for item_id, _ in session)) for session in valid_sessions]
        valid_context_indices_list = [
            [valid_candidate_items.index(item_id) for item_id in session if item_id in valid_candidate_items]
            for session in valid_context_indices_list
        ]
        valid_candidate_item_indices = np.array([valid_item_index_dict[item_id] for item_id in valid_candidate_items],
                                                dtype=np.int32)
        valid_predictions_mask = np.zeros(shape=(len(valid_sessions), len(valid_candidate_items)), dtype=np.float32)
        for i, context_indices in enumerate(valid_context_indices_list):
            valid_predictions_mask[i, context_indices] = -1000000.0

    valid_dataset = ValidDataset(
        args=args,
        valid_sessions=valid_sessions,
        valid_candidate_items=valid_candidate_items,
        valid_item_index_dict=valid_item_index_dict,
        metadata=metadata[len(MONTH_LIST)],
        metadata2m=metadata_2m[len(MONTH_LIST)],
        metadataall=metadata_all[len(MONTH_LIST)],
        metadatacurm=metadata_curm[len(MONTH_LIST)],
        cooccurrence=cooccurrence[len(MONTH_LIST)]
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=submission_args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: valid_collate_fn(batch=batch, candidate_item_indices=valid_candidate_item_indices),
        pin_memory=True,
        drop_last=False
    )

    model = Model(args=args)
    model.load_state_dict(torch.load(f'{MODEL_DIR}/{submission_args.model_name}.bin'))
    model.cuda()
    model.eval()

    with torch.no_grad(), torch.cuda.amp.autocast():
        valid_predictions = []
        item_encoded = model.item_encoder(
            item_id_batch=valid_item_id_batch,
            item_fixed_len_features_batch=valid_item_fixed_len_features_batch,
            item_var_len_features_batch=valid_item_var_len_features_batch,
            item_var_len_features_offsets_batch=valid_item_var_len_features_offsets_batch
        )
        for context_item_indices_batch, month_index_batch, dense_features_batch in tqdm(valid_dataloader):
            context_item_indices_batch = context_item_indices_batch.cuda()
            month_index_batch = month_index_batch.cuda()
            dense_features_batch = dense_features_batch.cuda()
            prediction_batch = model.predict(
                item_encoded=item_encoded,
                item_indices_batch=context_item_indices_batch,
                month_index_batch=month_index_batch,
                dense_features_batch=dense_features_batch
            )
            valid_predictions.append(prediction_batch.cpu().numpy())
        valid_predictions = np.concatenate(valid_predictions, axis=0).reshape(len(valid_sessions), len(valid_candidate_items))
        valid_predictions = torch.from_numpy(valid_predictions).cuda()
        valid_predictions += torch.from_numpy(valid_predictions_mask).cuda()
        np.save(f'{SUBMISSION_DIR}/{submission_args.model_name}_predictions.npy', valid_predictions.cpu().numpy())


if __name__ == '__main__':
    main()
