import random

import numpy as np
import torch
from torch.utils.data import Dataset

from config import MONTH_LIST


def extract_features(
        context_item_ids,
        candidate_item_ids,
        metadata,
        cooccurrence
):
    context_len = len(context_item_ids)
    purchase_count_normalized_1_month = metadata['purchase_count_normalized_1_month']
    context_purchase_count_normalized_1_month = [purchase_count_normalized_1_month.get(item_id, 0.0) for item_id in context_item_ids]
    context_purchase_count_normalized_1_month_mean = sum(context_purchase_count_normalized_1_month) / context_len
    impression_count_normalized_1_month = metadata['impression_count_normalized_1_month']
    context_impression_count_normalized_1_month = [impression_count_normalized_1_month.get(item_id, 0.0) for item_id in context_item_ids]
    context_impression_count_normalized_1_month_mean = sum(context_impression_count_normalized_1_month) / context_len

    context_features = [
        context_len / 100.0
    ]

    dense_features = []

    for target_item_id in candidate_item_ids:
        purchase_cooccurrence_row_normalized = [cooccurrence['purchase_cooccurrence_row_normalized'][target_item_id, item_id] for item_id in context_item_ids]
        purchase_cooccurrence_col_normalized = [cooccurrence['purchase_cooccurrence_col_normalized'][target_item_id, item_id] for item_id in context_item_ids]
        impression_cooccurrence_row_normalized = [cooccurrence['impression_cooccurrence_row_normalized'][target_item_id, item_id] for item_id in context_item_ids]
        impression_cooccurrence_col_normalized = [cooccurrence['impression_cooccurrence_col_normalized'][target_item_id, item_id] for item_id in context_item_ids]
        dense_features.append([
                                  max(purchase_cooccurrence_row_normalized),
                                  sum(purchase_cooccurrence_row_normalized) / context_len,
                                  purchase_cooccurrence_row_normalized[0],
                                  max(purchase_cooccurrence_col_normalized),
                                  sum(purchase_cooccurrence_col_normalized) / context_len,
                                  purchase_cooccurrence_col_normalized[0],
                                  max(impression_cooccurrence_row_normalized),
                                  sum(impression_cooccurrence_row_normalized) / context_len,
                                  impression_cooccurrence_row_normalized[0],
                                  max(impression_cooccurrence_col_normalized),
                                  sum(impression_cooccurrence_col_normalized) / context_len,
                                  impression_cooccurrence_col_normalized[0],
                                  purchase_count_normalized_1_month.get(target_item_id, 0.0) - context_purchase_count_normalized_1_month_mean,
                                  impression_count_normalized_1_month.get(target_item_id, 0.0) - context_impression_count_normalized_1_month_mean
                              ] + context_features)

    return dense_features

def extract_featuresall(
        context_item_ids,
        candidate_item_ids,
        metadata,
        metadata2m,
        metadataall,
        metadatacurm,
        cooccurrence
):
    context_len = len(context_item_ids)

    purchase_count_normalized_1_month = metadata['purchase_count_normalized_1_month']
    context_purchase_count_normalized_1_month = [purchase_count_normalized_1_month.get(item_id, 0.0) for item_id in context_item_ids]
    context_purchase_count_normalized_1_month_mean = sum(context_purchase_count_normalized_1_month) / context_len
    impression_count_normalized_1_month = metadata['impression_count_normalized_1_month']
    context_impression_count_normalized_1_month = [impression_count_normalized_1_month.get(item_id, 0.0) for item_id in context_item_ids]
    context_impression_count_normalized_1_month_mean = sum(context_impression_count_normalized_1_month) / context_len

    purchase_count_normalized_2_month = metadata2m['purchase_count_normalized_1_month']
    context_purchase_count_normalized_2_month = [purchase_count_normalized_2_month.get(item_id, 0.0) for item_id in context_item_ids]
    context_purchase_count_normalized_2_month_mean = sum(context_purchase_count_normalized_2_month) / context_len
    impression_count_normalized_2_month = metadata2m['impression_count_normalized_1_month']
    context_impression_count_normalized_2_month = [impression_count_normalized_2_month.get(item_id, 0.0) for item_id in context_item_ids]
    context_impression_count_normalized_2_month_mean = sum(context_impression_count_normalized_2_month) / context_len

    purchase_count_normalized_all_month = metadataall['purchase_count_normalized_1_month']
    context_purchase_count_normalized_all_month = [purchase_count_normalized_all_month.get(item_id, 0.0) for item_id in context_item_ids]
    context_purchase_count_normalized_all_month_mean = sum(context_purchase_count_normalized_all_month) / context_len
    impression_count_normalized_all_month = metadataall['impression_count_normalized_1_month']
    context_impression_count_normalized_all_month = [impression_count_normalized_all_month.get(item_id, 0.0) for item_id in context_item_ids]
    context_impression_count_normalized_all_month_mean = sum(context_impression_count_normalized_all_month) / context_len

    purchase_count_normalized_cur_month = metadatacurm['purchase_count_normalized_1_month']
    context_purchase_count_normalized_cur_month = [purchase_count_normalized_cur_month.get(item_id, 0.0) for item_id in context_item_ids]
    context_purchase_count_normalized_cur_month_mean = sum(context_purchase_count_normalized_cur_month) / context_len
    impression_count_normalized_cur_month = metadatacurm['impression_count_normalized_1_month']
    context_impression_count_normalized_cur_month = [impression_count_normalized_cur_month.get(item_id, 0.0) for item_id in context_item_ids]
    context_impression_count_normalized_cur_month_mean = sum(context_impression_count_normalized_cur_month) / context_len

    context_features = [
        context_len / 100.0
    ]

    dense_features = []

    for target_item_id in candidate_item_ids:
        purchase_cooccurrence_row_normalized = [cooccurrence['purchase_cooccurrence_row_normalized'][target_item_id, item_id] for item_id in context_item_ids]
        purchase_cooccurrence_col_normalized = [cooccurrence['purchase_cooccurrence_col_normalized'][target_item_id, item_id] for item_id in context_item_ids]
        impression_cooccurrence_row_normalized = [cooccurrence['impression_cooccurrence_row_normalized'][target_item_id, item_id] for item_id in context_item_ids]
        impression_cooccurrence_col_normalized = [cooccurrence['impression_cooccurrence_col_normalized'][target_item_id, item_id] for item_id in context_item_ids]
        dense_features.append([
                                  max(purchase_cooccurrence_row_normalized),
                                  sum(purchase_cooccurrence_row_normalized) / context_len,
                                  purchase_cooccurrence_row_normalized[0],
                                  max(purchase_cooccurrence_col_normalized),
                                  sum(purchase_cooccurrence_col_normalized) / context_len,
                                  purchase_cooccurrence_col_normalized[0],
                                  max(impression_cooccurrence_row_normalized),
                                  sum(impression_cooccurrence_row_normalized) / context_len,
                                  impression_cooccurrence_row_normalized[0],
                                  max(impression_cooccurrence_col_normalized),
                                  sum(impression_cooccurrence_col_normalized) / context_len,
                                  impression_cooccurrence_col_normalized[0],
                                  purchase_count_normalized_1_month.get(target_item_id, 0.0) - context_purchase_count_normalized_1_month_mean,
                                  impression_count_normalized_1_month.get(target_item_id, 0.0) - context_impression_count_normalized_1_month_mean,
                                  purchase_count_normalized_2_month.get(target_item_id, 0.0) - context_purchase_count_normalized_2_month_mean,
                                  impression_count_normalized_2_month.get(target_item_id, 0.0) - context_impression_count_normalized_2_month_mean,
                                  purchase_count_normalized_all_month.get(target_item_id, 0.0) - context_purchase_count_normalized_all_month_mean,
                                  impression_count_normalized_all_month.get(target_item_id, 0.0) - context_impression_count_normalized_all_month_mean,
                                  purchase_count_normalized_cur_month.get(target_item_id, 0.0) - context_purchase_count_normalized_cur_month_mean,
                                  impression_count_normalized_cur_month.get(target_item_id, 0.0) - context_impression_count_normalized_cur_month_mean,
                              ] + context_features)

    return dense_features


class TrainDataset(Dataset):
    def __init__(
            self,
            args,
            train_sessions,
            train_item_index_dict,
            train_candidate_items,
            train_impression_items,
            metadata,
            cooccurrence
    ):
        self.args = args
        self.train_sessions = train_sessions
        self.train_item_index_dict = train_item_index_dict
        self.train_candidate_items = train_candidate_items
        self.train_impression_items = train_impression_items
        self.metadata = metadata
        self.cooccurrence = cooccurrence

    def __getitem__(self, index):
        session = self.train_sessions[index]
        context, target = session[: -1], session[-1]
        session_start_time = context[0][1]
        month_index = (session_start_time.year - 2020) * 12 + session_start_time.month - 1
        context = context[-self.args.context_max_len:][::-1]
        context_item_ids, context_times = zip(*context)
        target_item_id, _ = target
        context_item_indices = [self.train_item_index_dict[item_id] for item_id in context_item_ids]
        context_item_mask = list(np.random.choice(
            [True, False],
            size=len(context_item_ids),
            p=[self.args.mask_probability, 1.0 - self.args.mask_probability]
        )) if len(context_item_ids) > 1 else [False]
        masked_item_indices = [1] + [
            1 if context_item_mask[i] else context_item_indices[i]
            for i in range(len(context_item_indices))
        ]
        target_item_index = self.train_item_index_dict[target_item_id]
        target_item_indices = [target_item_index] + context_item_indices
        candidate_item_ids = self.train_candidate_items[month_index] - set(context_item_ids)
        candidate_item_indices = [self.train_item_index_dict[item_id] for item_id in candidate_item_ids]
        impression_item_indices = self.train_impression_items[month_index]
        impression_item_indices = [self.train_item_index_dict[item_id] for item_id in impression_item_indices]
        negative_item_ids = random.sample(candidate_item_ids - {target_item_id}, k=self.args.num_negatives)
        negative_item_indices = [
            [self.train_item_index_dict[item_id]] + context_item_indices
            for item_id in negative_item_ids
        ]
        dense_features = extract_features(
            context_item_ids=context_item_ids,
            candidate_item_ids=[target_item_id] + negative_item_ids,
            metadata=self.metadata[month_index],
            cooccurrence=self.cooccurrence[month_index]
        )
        return (
            masked_item_indices,
            target_item_indices,
            negative_item_indices,
            month_index,
            candidate_item_indices,
            impression_item_indices,
            dense_features
        )

    def __len__(self):
        return len(self.train_sessions)


class ValidDataset(Dataset):
    def __init__(
            self,
            args,
            valid_sessions,
            valid_candidate_items,
            valid_item_index_dict,
            metadata,
            cooccurrence
    ):
        self.args = args
        self.valid_sessions = valid_sessions
        self.valid_candidate_items = valid_candidate_items
        self.valid_item_index_dict = valid_item_index_dict
        self.metadata = metadata
        self.cooccurrence = cooccurrence

    def __getitem__(self, index):
        session = self.valid_sessions[index]
        context, _ = session[: -1], session[-1]
        context = context[-self.args.context_max_len:][::-1]
        context_item_ids, context_times = zip(*context)
        context_item_indices = [self.valid_item_index_dict[item_id] for item_id in context_item_ids]
        month_index = len(MONTH_LIST) - 1
        dense_features = extract_features(
            context_item_ids=context_item_ids,
            candidate_item_ids=self.valid_candidate_items,
            metadata=self.metadata,
            cooccurrence=self.cooccurrence
        )
        return (
            context_item_indices,
            month_index,
            dense_features
        )

    def __len__(self):
        return len(self.valid_sessions)


def valid_collate_fn(batch, candidate_item_indices):
    (
        context_item_indices_list,
        month_index_list,
        dense_features_list
    ) = zip(*batch)
    context_max_len = max(len(context_item_indices) for context_item_indices in context_item_indices_list)
    context_item_indices_batch = np.array([
        context_item_indices + [0] * (context_max_len - len(context_item_indices))
        for context_item_indices in context_item_indices_list
    ], dtype=np.int64)
    context_item_indices_batch = np.concatenate([
        np.expand_dims(candidate_item_indices, axis=(0, 2)).repeat(len(context_item_indices_list), axis=0),
        np.expand_dims(context_item_indices_batch, axis=1).repeat(len(candidate_item_indices), axis=1)
    ], axis=2)
    context_item_indices_batch = torch.from_numpy(context_item_indices_batch).flatten(0, 1)
    month_index_batch = torch.from_numpy(np.array(month_index_list, dtype=np.int64).repeat(len(candidate_item_indices)))
    dense_features_batch = torch.FloatTensor(dense_features_list)
    return (
        context_item_indices_batch,
        month_index_batch,
        dense_features_batch
    )
