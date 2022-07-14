import torch
from torch.utils.data import Dataset
import numpy as np


class SecondStageFirst(Dataset):
    def __init__(self, data_x_in):
        self.X_in = data_x_in

    def __getitem__(self, index):
        x_in = self.X_in[index]
        return torch.FloatTensor(x_in)

    def __len__(self):
        return self.X_in.shape[0]


class SecondStageTest(Dataset):
    def __init__(self, item_features, all_session_feature, first_stage_predictions, first_stage_scores,
                 item_bpr_embed=None, itemknn=None, session_feat=None):
        self.item_content = item_features
        self.all_session_feature = all_session_feature
        self.first_stage_predictions = first_stage_predictions
        self.first_stage_scores = first_stage_scores
        self.item_bpr_embed = item_bpr_embed
        self.itemknn = itemknn
        self.session_feat = session_feat

    def __getitem__(self, index):
        candidate_content = self.item_content[self.first_stage_predictions[index]]  # 200, 977
        candidate_scores = self.first_stage_scores[index].reshape(-1, self.first_stage_scores.shape[-1])
        candidate_feature = np.concatenate((candidate_content, candidate_scores), axis=1)
        if self.session_feat is not None:
            session_feature = np.concatenate((self.all_session_feature[index], self.session_feat[index]))
        else:
            session_feature = self.all_session_feature[index]
        if self.itemknn is not None:
            candidate_knn = self.itemknn[self.first_stage_predictions[index]]
            candidate_feature = np.concatenate((candidate_feature, candidate_knn), axis=1)
        if self.item_bpr_embed is not None:
            candidate_bpr_embed = self.item_bpr_embed[self.first_stage_predictions[index]]
            candidate_feature = np.concatenate((candidate_feature, candidate_bpr_embed), axis=1)
        return torch.FloatTensor(session_feature), \
               torch.FloatTensor(candidate_feature), \
               torch.LongTensor(self.first_stage_predictions[index])

    def __len__(self):
        return self.all_session_feature.shape[0]
