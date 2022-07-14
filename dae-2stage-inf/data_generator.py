import numpy as np
import os
import pickle
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale, minmax_scale, maxabs_scale, robust_scale
import math
from tqdm import tqdm
from config import *


def load(filename, directory=None):
    if directory:
        return pickle.load(open(os.path.join(directory, filename), "rb"))
    else:
        return pickle.load(open(filename, "rb"))


def save(obj, filename, directory=None):
    if directory:
        with open(os.path.join(directory, filename), "wb+") as file_out:
            pickle.dump(obj, file_out)
    else:
        with open(filename, "wb+") as file_out:
            pickle.dump(obj, file_out)


class Data(object):
    def __init__(self, args):
        self.args = args
        self.item_idx_to_id = load(CACHE_DIR + '/item_idx_to_id.pkl')
        self.item_id_to_idx = {v: k for k, v in self.item_idx_to_id.items()}

        self.item_features = self.process_content(load(CACHE_DIR + '/item_features.pkl'))
        self.num_items = self.item_features.shape[0]

        if self.args.mode == 'training':
            self.X_train, X_val_all, self.y_train, self.y_val, self.candidate_val = load(CACHE_DIR + '/train_set.pkl')

            if args.cut_train:
                cut = int(args.cut_train * len(self.X_train))
                self.X_train = self.X_train[cut:]
                self.y_train = self.y_train[cut:]

            if args.use_itemknn:
                self.itemknn_feature = self.process_knn_scores(eval(args.sample)[2])

            self.process_item()
            self.candidate_val = self.candidate_val.astype('int64')

            self.val_sess = pickle.load(open(CACHE_DIR + '/val_sess.pkl', 'rb'))
            self.sess_ids = []
            self.y_val = []
            self.X_val = []

            print('load zhaolin val')
            for i in range(len(self.val_sess)):
                sess_id, context_items, ytrue = self.val_sess[i][0], self.val_sess[i][1:-1], self.val_sess[i][-1]
                self.sess_ids.append(self.val_sess[i])
                self.X_val.append([self.item_id_to_idx[x] for x in context_items])
                self.y_val.append(self.item_id_to_idx[ytrue])

            self.X_val = self.process_val()
            self.sess_ids = load(CACHE_DIR + '/val_sess_id.pkl')

        elif self.args.mode == 'lb':
            self.X_train, X_val_all, self.y_train, y_val, _ = load(CACHE_DIR + '/train_set.pkl')
            self.X_train += X_val_all
            self.y_train = np.concatenate((self.y_train, y_val)).astype('int64')

            if args.cut_train:
                cut = int(args.cut_train * len(self.X_train))
                self.X_train = self.X_train[cut:]
                self.y_train = self.y_train[cut:]

            self.process_item()
            self.candidate_val = np.array(load(CACHE_DIR + '/candidate_items.pkl')).astype('int64')
            self.sess_ids, self.X_val = load(CACHE_DIR + '/leaderboard_set.pkl')
            self.X_val = self.process_val()
            self.item_idx_to_id = load(CACHE_DIR + '/item_idx_to_id.pkl')

            if args.use_itemknn:
                self.itemknn_feature = self.process_knn_scores(eval(args.sample)[2])

        elif self.args.mode == 'final':
            self.X_train, X_val_all, self.y_train, y_val, _ = load(CACHE_DIR + '/train_set.pkl')
            self.X_train += X_val_all
            self.y_train = np.concatenate((self.y_train, y_val)).astype('int64')

            self.process_item()
            self.candidate_val = np.array(load(CACHE_DIR + '/candidate_items.pkl')).astype('int64')
            self.sess_ids, self.X_val = load(CACHE_DIR + '/final_set.pkl')
            self.X_val = self.process_val()
            self.item_idx_to_id = load(CACHE_DIR + '/item_idx_to_id.pkl')

            if args.use_itemknn:
                self.itemknn_feature = self.process_knn_scores(eval(args.sample)[2])

    def process_content(self, raw_features):
        cat_val_to_idx, cat_to_idx = {}, {}
        idx_catval, idx_cat = 0, 0
        for i in raw_features:
            for j in i:
                feature = str(j[0]) + ',' + str(j[1])
                if str(j[0]) not in cat_to_idx:
                    cat_to_idx[str(j[0])] = idx_cat
                    idx_cat += 1
                if feature not in cat_val_to_idx:
                    cat_val_to_idx[feature] = idx_catval
                    idx_catval += 1
        row, col = [], []
        for i in range(len(raw_features)):
            for j in raw_features[i]:
                row.append(i)
                col.append(cat_val_to_idx[str(j[0]) + ',' + str(j[1])])
        item_feature = sp.csr_matrix((np.ones(len(row)), (row, col)),
                                     shape=(len(raw_features), len(cat_val_to_idx))).A

        if eval(self.args.sample)[0] > 0:
            threshold = int(len(item_feature) * eval(self.args.sample)[0])
            feature_mask = item_feature.sum(0) > threshold
            item_feature = item_feature[:, feature_mask]
            print("dropped item cat val with sample=" + str(eval(self.args.sample)[0]) + ", " + str(
                item_feature.shape[1]) + " left")

        if self.args.use_cat:
            row, col = [], []
            for i in range(len(raw_features)):
                for j in raw_features[i]:
                    row.append(i)
                    col.append(cat_to_idx[str(j[0])])
            item_cat_feature = sp.csr_matrix((np.ones(len(row)), (row, col)),
                                             shape=(len(raw_features), len(cat_to_idx))).A
            if eval(self.args.sample)[1] > 0:
                threshold = int(len(item_cat_feature) * eval(self.args.sample)[1])
                feature_mask = item_cat_feature.sum(0) > threshold
                item_cat_feature = item_cat_feature[:, feature_mask]
                print("dropped item cat with sample=" + str(eval(self.args.sample)[1]) + ", " + str(
                    item_cat_feature.shape[1]) + " left")
            item_feature = np.concatenate((item_feature, item_cat_feature), axis=1)

        self.onehot_item_feature = item_feature[:]

        item_feature = normalize(item_feature, norm=self.args.norm, axis=1)
        if self.args.col_norm == 'scale':
            item_feature = scale(item_feature, axis=0)
        elif self.args.col_norm == 'minmax_scale':
            item_feature = minmax_scale(item_feature, axis=0)
        elif self.args.col_norm == 'maxabs_scale':
            item_feature = maxabs_scale(item_feature, axis=0)
        elif self.args.col_norm == 'robust_scale':
            item_feature = robust_scale(item_feature, axis=0)
        return item_feature

    def process_knn_scores(self, sample):
        item_id_to_idx = {v: k for k, v in self.item_idx_to_id.items()}
        if self.args.mode == 'training':
            itemknn_sims = np.load(CACHE_DIR + '/sims.npy', allow_pickle=True).item()
        else:
            itemknn_sims = np.load(CACHE_DIR + '/sims_lb_500.npy', allow_pickle=True).item()
        row, col, data = [], [], []
        for item_id, df_val in tqdm(itemknn_sims.items()):
            feature = df_val[df_val > 0]
            for item, score in zip(feature.index, feature.values):
                if item in item_id_to_idx:
                    row.append(item_id_to_idx[item_id])
                    col.append(item_id_to_idx[item])
                    data.append(score)
        knn_scores = sp.csr_matrix((data, (row, col)), shape=(len(item_id_to_idx), 1 + max(col))).A

        if sample > 0:
            threshold = int(len(knn_scores) * sample)
            feature_mask = (knn_scores != 0).sum(0) > threshold
            knn_scores = knn_scores[:, feature_mask]
            print("dropped knn feature with sample=" + str(sample) + ", " + str(
                knn_scores.shape[1]) + " left")
        knn_scores = normalize(knn_scores, norm='l2', axis=1)

        if self.args.knn_col_norm == 'scale':
            print('processing knn with scale')
            knn_scores = scale(knn_scores, axis=0)
        elif self.args.knn_col_norm == 'minmax_scale':
            print('processing knn with minmax_scale')
            knn_scores = minmax_scale(knn_scores, axis=0)
        elif self.args.knn_col_norm == 'maxabs_scale':
            print('processing knn with maxabs_scale')
            knn_scores = maxabs_scale(knn_scores, axis=0)
        elif self.args.knn_col_norm == 'robust_scale':
            print('processing knn with robust_scale')
            knn_scores = robust_scale(knn_scores, axis=0)
        return knn_scores

    def process_item(self):
        row = []
        col = []
        for i in range(len(self.X_train)):
            row.append(i)
            col.append(self.y_train[i])
        matrix_3 = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(len(self.X_train), self.num_items))
        temp = np.sum(matrix_3, 0)
        self.hot_items = np.nonzero(temp)[1]
        self.cold_items = np.where(temp == 0)[1]

    def process_train(self):
        row = []
        col = []
        data = []

        if self.args.drop_val:
            val_index = np.arange(len(self.X_train) - 80000, len(self.X_train), 1)
            val_index_drop = np.random.choice(val_index, size=int(self.args.drop_val * 80000), replace=False)

            val_index_keep = [x for x in val_index if x not in val_index_drop]

        for i in range(len(self.X_train)):
            indexes = np.unique(self.X_train[i][::-1], return_index=True)[1]
            sess = [self.X_train[i][::-1][index] for index in sorted(indexes)]
            length = len(sess)
            if self.args.max_len:
                length = min(len(sess), self.args.max_len)
            for j in range(length):
                row.append(i)
                col.append(sess[j])
                data.append(1. / math.log(math.exp(self.args.time_pad * j) + 1))
        matrix_2 = sp.csr_matrix((data, (row, col)), shape=(len(self.X_train), self.num_items))
        row = []
        col = []
        for i in range(len(self.X_train)):
            row.append(i)
            col.append(self.y_train[i])

        matrix_3 = sp.csr_matrix((np.full(len(row), self.args.purchase_weight), (row, col)), \
                                 shape=(len(self.X_train), self.num_items))
        matrix_1 = matrix_3 + matrix_2

        self.X_in, self.X_out = matrix_2, matrix_1

        if self.args.drop_val:
            index_kept = np.arange(0, len(self.X_train) - 80000, 1).astype('int').tolist() + val_index_keep
            self.X_in = self.X_in[index_kept][:]
            self.X_out = self.X_out[index_kept][:]
            self.y_train = self.y_train[index_kept][:]

    def process_val(self):
        row = []
        col = []
        data = []

        for i in range(len(self.X_val)):
            indexes = np.unique(self.X_val[i][::-1], return_index=True)[1]
            sess = [self.X_val[i][::-1][index] for index in sorted(indexes)]
            length = len(sess)
            for j in range(length):
                row.append(i)
                col.append(sess[j])
                data.append(1. / math.log(math.exp(self.args.time_pad * j) + 1))
        val_matrix = sp.csr_matrix((data, (row, col)), shape=(len(self.X_val), self.num_items))
        return val_matrix.toarray()
