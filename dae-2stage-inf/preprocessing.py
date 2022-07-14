import pandas as pd
import numpy as np
import os
import pickle
from config import *


def save(obj, filename, directory=None):
    if directory:
        with open(os.path.join(directory, filename), "wb+") as file_out:
            pickle.dump(obj, file_out)
    else:
        with open(filename, "wb+") as file_out:
            pickle.dump(obj, file_out)


if __name__ == '__main__':

    # re-index items and extract features
    features = pd.read_csv(DATA_DIR + '/item_features.csv').to_numpy()

    all_item = np.unique(features[:, 0])
    item_idx = dict(zip(all_item, range(len(all_item))))
    idx_to_id = dict(zip(range(len(all_item)), all_item))
    item_features = [[] for i in range(len(all_item))]

    for i in features:
        item_features[item_idx[i[0]]].append([i[1], i[2]])
    save(item_features, CACHE_DIR + '/item_features.pkl')
    save(idx_to_id, CACHE_DIR + '/item_idx_to_id.pkl')
    print('Number of items:', len(all_item))

    # re-index items and split
    train_s = pd.read_csv(DATA_DIR + '/train_sessions.csv').sort_values(['session_id', 'date']).to_numpy()
    train_p = pd.read_csv(DATA_DIR + '/train_purchases.csv').sort_values(['date']).to_numpy()

    sess_idx = dict(zip(train_p[:, 0], range(len(train_p[:, 0]))))
    X = [[] for i in range(len(train_p[:, 0]))]
    y = train_p[:, 1]

    for i in train_s:
        X[sess_idx[i[0]]].append(item_idx[i[1]])
    for i in range(len(y)):
        y[i] = item_idx[y[i]]

    train_size = len(train_p) - (train_p[:, 2] >= '2021-05').sum()

    train_s_timestamp = train_s[:, -1].astype('datetime64', copy=False).astype('int')
    X_time = [[] for i in range(len(X))]
    X_month = [[] for i in range(len(X))]
    for i in range(len(train_s)):
        X_time[sess_idx[train_s[i][0]]].append(train_s_timestamp[i])
        X_month[sess_idx[train_s[i][0]]].append(train_s[i][-1][:7])

    X_month = [x[0] for x in X_month]
    X_train_ts, X_test_ts = X_time[:train_size], X_time[train_size:]
    save((X_train_ts, X_test_ts), CACHE_DIR + '/train_ts.pkl')
    # X_train_month, X_test_month = X_month[:train_size], X_month[train_size:]
    # save((X_train_month, X_test_month), CACHE_DIR + 'train_month.pkl')

    train_sess_ids = [*sess_idx.keys()]
    save((train_sess_ids[:train_size], train_sess_ids[train_size:]), CACHE_DIR + '/train_sess_id.pkl')

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    candidate_test = np.unique(y_test)
    save((X_train, X_test, y_train, y_test, candidate_test), CACHE_DIR + '/train_set.pkl')

    save(train_p[train_size:][:, 0], CACHE_DIR + '/val_sess_id.pkl')

    print('Training sessions:', len(X_train))
    print('Validation sessions:', len(X_test))
    print('Number of candidates for validation:', len(candidate_test))

    # re-index candidates
    candidate = pd.read_csv(DATA_DIR + '/candidate_items.csv').to_numpy().squeeze()
    new_candidates = []

    for i in candidate:
        new_candidates.append(item_idx[i])
    save(new_candidates, CACHE_DIR + '/candidate_items.pkl')
    print('Number of candidates for LB and final:', len(new_candidates))

    # re-index
    lb_s = pd.read_csv(DATA_DIR + '/test_leaderboard_sessions.csv').sort_values(['date']).to_numpy()
    indexes = np.unique(lb_s[:, 0], return_index=True)[1]
    lb_sess = np.array([lb_s[:, 0][index] for index in sorted(indexes)])

    save(lb_sess, CACHE_DIR + '/lb_sess_id.pkl')

    sess_idx = dict(zip(lb_sess, range(len(lb_sess))))
    lb_X = [[] for i in range(len(lb_sess))]

    for i in lb_s:
        lb_X[sess_idx[i[0]]].append(item_idx[i[1]])
    save((lb_sess, lb_X), CACHE_DIR + '/leaderboard_set.pkl')
    print('Leaderboard sessions:', len(lb_sess))

    final_s = pd.read_csv(DATA_DIR + '/test_final_sessions.csv').sort_values(['date']).to_numpy()
    indexes = np.unique(final_s[:, 0], return_index=True)[1]
    final_sess = np.array([final_s[:, 0][index] for index in sorted(indexes)])
    sess_idx = dict(zip(final_sess, range(len(final_sess))))
    final_X = [[] for i in range(len(final_sess))]

    for i in final_s:
        final_X[sess_idx[i[0]]].append(item_idx[i[1]])
    save((final_sess, final_X), CACHE_DIR + '/final_set.pkl')
    print('Final sessions:', len(final_sess))
