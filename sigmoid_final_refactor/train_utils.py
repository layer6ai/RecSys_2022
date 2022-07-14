from collections import OrderedDict
from copy import deepcopy

import numpy as np

from config import NUM_FEATURE_CATEGORY_ID, VAR_LEN_FEATURE_CATEGORY_ID_LIST


def get_item_fixed_len_feature_category_ids():
    return [
        feature_category_id for feature_category_id in range(1, NUM_FEATURE_CATEGORY_ID + 1)
        if feature_category_id != 27 and feature_category_id not in VAR_LEN_FEATURE_CATEGORY_ID_LIST
    ]


def build_item_index_dict(item_counts):
    item_index_dict = OrderedDict([('[PAD]', 0), ('[MASK]', 1)])
    for item_id in item_counts:
        item_index_dict[item_id] = len(item_index_dict)
    return item_index_dict


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
