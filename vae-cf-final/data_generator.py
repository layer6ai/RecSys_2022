import numpy as np
import os
import pickle
import scipy.sparse as sp
from sklearn.preprocessing import normalize


def load(filename, directory=None):
    if directory:
        return pickle.load(open(os.path.join(directory, filename), "rb"))
    else:
        return pickle.load(open(filename, "rb"))

class Data(object):
    def __init__(self):

        item_cat = load('./data/item_features.pkl')
        self.num_items = len(item_cat)

        _, self.X_train, X_val_all, self.X_t_train, X_t_val_all, self.X_m_train, X_m_val, \
            self.y_train, y_val, _ = load('./data/train_set.pkl')
        self.X_train += X_val_all
        self.X_t_train += X_t_val_all
        self.X_m_train += X_m_val
        self.y_train = np.concatenate((self.y_train, y_val)).astype('int64')
        self.candidate_val = np.array(load('./data/candidate_items.pkl')).astype('int64')
        self.num_month = max(self.X_m_train)+1

        self.sess_ids, self.X_val, self.X_t_val = load('./data/leaderboard_set.pkl')
        test_sess_ids, test_X, test_X_t = load('./data/final_set.pkl')
        self.len_lb, self.len_final = len(self.sess_ids), len(test_sess_ids)
        self.sess_ids = np.concatenate((self.sess_ids, test_sess_ids))
        self.X_val += test_X
        self.X_t_val += test_X_t
        self.process_val()

        self.item_idx_to_id = load('./data/item_idx_to_id.pkl')

        self.item_features, self.cooccurence = self.process_features(item_cat)

    def process_val(self):

        self.X_m_val = [self.num_month-1 for i in range(len(self.X_val))]
        self.val_sess_len = [[] for i in range(10)]

        row, col, data = [], [], []
        row_l, col_l = [], []
        for i in range(len(self.X_val)):
 
            indexes = np.unique(self.X_val[i][::-1], return_index=True)[1]
            sess = [self.X_val[i][::-1][index] for index in sorted(indexes)]

            order_pad = 1./np.log(np.exp(1*np.arange(len(sess)))+1)

            self.val_sess_len[min(9, len(sess)-1)].append(i)
            row_l.append(i)
            col_l.append(sess[0])

            for j in range(len(sess)):
                row.append(i)
                col.append(sess[j])
                data.append(order_pad[j])

        self.X_val = sp.csr_matrix((data, (row, col)), \
                                    shape=(len(self.X_val), self.num_items))
        self.X_val = normalize(self.X_val, norm='l2', axis=1)

    def process_features(self, raw_features):
        feature_to_idx = {}
        idx = 0
        for i in raw_features:
            for j in i:
                feature = str(j[0]) + ',' + str(j[1])
                if feature not in feature_to_idx:
                    feature_to_idx[feature] = idx
                    idx += 1
        
        row, col = [], []
        for i in range(len(raw_features)):
            for j in raw_features[i]:
                row.append(i)
                col.append(feature_to_idx[str(j[0]) + ',' + str(j[1])])

        item_features = sp.csr_matrix((np.ones(len(row)), (row, col)), \
                                    shape=(len(raw_features), len(feature_to_idx)))
        item_features = normalize(item_features, norm='l2', axis=1)

        # cooccurrence matrix for last month
        row, col = [], []
        for i in range(len(self.X_m_train)):
            if self.X_m_train[i] == self.num_month-1:
                for j in range(len(self.X_train[i])-1):
                    row.append(self.X_train[i][j])
                    col.append(self.X_train[i][j+1])
                    row.append(self.X_train[i][j+1])
                    col.append(self.X_train[i][j])
                row.append(self.X_train[i][-1])
                col.append(self.y_train[i])
                row.append(self.y_train[i])
                col.append(self.X_train[i][-1])
        cooccurence = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(self.num_items, self.num_items)).todok()

        rowsum = np.array(cooccurence.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        cooccurence = d_mat.dot(cooccurence)
        cooccurence = cooccurence.dot(d_mat)
        cooccurence = cooccurence.tocsr()

        return item_features, cooccurence
