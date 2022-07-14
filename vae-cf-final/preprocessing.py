import pandas as pd
import numpy as np
import os
import pickle


def save(obj, filename, directory=None):
    if directory:
        with open(os.path.join(directory, filename), "wb+") as file_out:
            pickle.dump(obj, file_out)
    else:
        with open(filename, "wb+") as file_out:
            pickle.dump(obj, file_out)


def preprocessing():

    # re-index items and extract features
    features = pd.read_csv('./data/item_features.csv').to_numpy()

    all_item = np.unique(features[:, 0])
    item_idx = dict(zip(all_item, range(len(all_item))))
    idx_to_id = dict(zip(range(len(all_item)), all_item))
    item_features = [[] for i in range(len(all_item))]

    for i in features:
        item_features[item_idx[i[0]]].append([i[1], i[2]])
    save(item_features, 'item_features.pkl', './data/')
    save(idx_to_id, 'item_idx_to_id.pkl', './data/')
    print('Number of items:', len(all_item))

    # re-index items and split
    train_s = pd.read_csv('./data/train_sessions.csv').sort_values(['session_id', 'date']).to_numpy()
    train_p = pd.read_csv('./data/train_purchases.csv').sort_values(['date']).to_numpy()

    sess_idx = dict(zip(train_p[:, 0], range(len(train_p[:, 0]))))
    X = [[] for i in range(len(train_p[:, 0]))]
    X_t = [[] for i in range(len(train_p[:, 0]))]
    X_m = []
    y = train_p[:, 1]

    for i in train_s:
        X[sess_idx[i[0]]].append(item_idx[i[1]])
        X_t[sess_idx[i[0]]].append(i[2])
    for i in range(len(y)):
        y[i] = item_idx[y[i]]

    for i in train_p[:, 2]:
        if i < '2021':
            X_m.append(int(i[5:7])-1)
        else:
            X_m.append(int(i[5:7])+11)

    for i in range(len(X_t)):
        for j in range(len(X_t[i])-1):
            X_t[i][j] = (np.datetime64(X_t[i][-1])-np.datetime64(X_t[i][j])).astype(int)
        X_t[i][-1] = 0

    train_size = len(train_p)-(train_p[:, 2]>='2021-05').sum()

    sess_id = train_p[train_size:, 0]
    X_train, X_test = X[:train_size], X[train_size:]
    X_t_train, X_t_test = X_t[:train_size], X_t[train_size:]
    X_m_train, X_m_test = X_m[:train_size], X_m[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    candidate_test = np.unique(y_test)
    save((sess_id, X_train, X_test, X_t_train, X_t_test, X_m_train, X_m_test, y_train, y_test, candidate_test), 'train_set.pkl', './data/')
    print('Training sessions:', len(X_train))
    print('Validation sessions:', len(X_test))
    print('Number of candidates for validation:', len(candidate_test))

    # re-index candidates
    candidate = pd.read_csv('./data/candidate_items.csv').to_numpy().squeeze()
    new_candidates = []

    for i in candidate:
        new_candidates.append(item_idx[i])
    save(new_candidates, 'candidate_items.pkl', './data/')
    print('Number of candidates for LB and final:', len(new_candidates))

    # re-index
    lb_s = pd.read_csv('./data/test_leaderboard_sessions.csv').sort_values(['date']).to_numpy()
    indexes = np.unique(lb_s[:, 0], return_index=True)[1]
    lb_sess = np.array([lb_s[:, 0][index] for index in sorted(indexes)])
    sess_idx = dict(zip(lb_sess, range(len(lb_sess))))
    lb_X = [[] for i in range(len(lb_sess))]
    lb_X_t = [[] for i in range(len(lb_sess))]

    for i in lb_s:
        lb_X[sess_idx[i[0]]].append(item_idx[i[1]])
        lb_X_t[sess_idx[i[0]]].append(i[2])
    
    for i in range(len(lb_X_t)):
        for j in range(len(lb_X_t[i])-1):
            lb_X_t[i][j] = (np.datetime64(lb_X_t[i][-1])-np.datetime64(lb_X_t[i][j])).astype(int)
        lb_X_t[i][-1] = 0

    save((lb_sess, lb_X, lb_X_t), 'leaderboard_set.pkl', './data/')
    print('Leaderboard sessions:', len(lb_sess))

    final_s = pd.read_csv('./data/test_final_sessions.csv').sort_values(['date']).to_numpy()
    indexes = np.unique(final_s[:, 0], return_index=True)[1]
    final_sess = np.array([final_s[:, 0][index] for index in sorted(indexes)])
    sess_idx = dict(zip(final_sess, range(len(final_sess))))
    final_X = [[] for i in range(len(final_sess))]
    final_X_t = [[] for i in range(len(final_sess))]

    for i in final_s:
        final_X[sess_idx[i[0]]].append(item_idx[i[1]])
        final_X_t[sess_idx[i[0]]].append(i[2])

    for i in range(len(final_X_t)):
        for j in range(len(final_X_t[i])-1):
            final_X_t[i][j] = (np.datetime64(final_X_t[i][-1])-np.datetime64(final_X_t[i][j])).astype(int)
        final_X_t[i][-1] = 0

    save((final_sess, final_X, final_X_t), 'final_set.pkl', './data/')
    print('Final sessions:', len(final_sess))

