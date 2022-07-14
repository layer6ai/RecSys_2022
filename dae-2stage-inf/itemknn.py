import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse
from config import *


def exp_time_func(x, t0=100):
    lmda = - t0 / np.log(0.5)
    return np.exp((-1 / lmda) * x)


def logit_time_func(x, k, t0):
    '''k: drop rate, smaller k means faster drop
       t0: when the output = 0.5
    '''
    return 1 / (1 + np.exp((-1 / k) * (-(x - t0))))


class ItemKNN:
    def __init__(self, n_sims=100, lmbd=20, alpha=0.5, t0=100, k=0, bw=1,
                 session_key='session_id', item_key='item_id', time_key='date'):
        self.n_sims = n_sims
        self.lmbd = lmbd
        self.alpha = alpha
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.bw = bw
        self.t0 = t0
        self.k = k
        print("lmbd=%d, alpha=%.2f, t0=%d, k=%s, bw=%.2f" % (self.lmbd, self.alpha, self.t0, self.k, self.bw))

    def fit(self, data):
        print("fitting itemknn... ")
        data.set_index(np.arange(len(data)), inplace=True)
        max_day = datetime.strptime(data.date.max().split()[0], '%Y-%m-%d')
        itemids = data[self.item_key].unique()
        n_items = len(itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': np.arange(len(itemids))}),
                        on=self.item_key, how='inner')
        sessionids = data[self.session_key].unique()
        data = pd.merge(data, pd.DataFrame({self.session_key: sessionids, 'SessionIdx': np.arange(len(sessionids))}),
                        on=self.session_key, how='inner')
        session_day = data[['SessionIdx', 'date']].copy()
        session_day['day'] = session_day['date'].apply(lambda x: x[:10])
        session_day = session_day[['SessionIdx', 'day']].drop_duplicates()
        session_day_dict = session_day.set_index("SessionIdx").to_dict()['day']
        supp = data.groupby('SessionIdx').size()
        session_offsets = np.zeros(len(supp) + 1, dtype=np.int32)
        session_offsets[1:] = supp.cumsum()
        index_by_sessions = data.sort_values(['SessionIdx', self.time_key]).index.values
        supp = data.groupby('ItemIdx').size()
        item_offsets = np.zeros(n_items + 1, dtype=np.int32)
        item_offsets[1:] = supp.cumsum()
        index_by_items = data.sort_values(['ItemIdx', self.time_key]).index.values
        self.sims = dict()
        session_buy = data[data.type == 'buy'].set_index("SessionIdx").to_dict()['ItemIdx']
        self.sims_full = dict()

        for i in range(n_items):
            iarray = np.zeros(n_items)
            start = item_offsets[i]
            tmp_session_set = set()
            end = item_offsets[i + 1]  # all sessions included this item
            for e in index_by_items[start:end]:
                uidx = data.SessionIdx.values[e]  # the session id that also has this item
                ustart = session_offsets[uidx]
                uend = session_offsets[uidx + 1]
                user_events = index_by_sessions[ustart:uend]  # all items in that session
                if uidx not in tmp_session_set:
                    tmp_session_set.add(uidx)
                else:
                    continue
                bought_item = session_buy[uidx]
                curr_day = datetime.strptime(session_day_dict[uidx], '%Y-%m-%d')
                if self.k != 0:
                    score = logit_time_func(x=(max_day - curr_day).days, k=self.k, t0=self.t0)
                elif self.t0 != 0:
                    score = exp_time_func(x=(max_day - curr_day).days, t0=self.t0)
                else:
                    score = 1
                iarray[data.ItemIdx.values[user_events]] += score
                iarray[bought_item] *= self.bw
            iarray[i] = 0  # set itself to 0
            norm = np.power((supp[i] + self.lmbd), self.alpha) * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
            norm[norm == 0] = 1
            iarray = iarray / norm
            indices = np.argsort(iarray)[-self.n_sims:][::-1]
            self.sims_full[itemids[i]] = iarray
            self.sims[itemids[i]] = pd.Series(data=iarray[indices], index=itemids[indices])


def gen_knn_mat(train_sessions):
    print('\nsaving item knn with all train session...')
    itemknn = ItemKNN(lmbd=args.lmbd, alpha=args.alpha, t0=args.t0, k=args.k, bw=args.bw, n_sims=args.n_sims)
    itemknn.fit(train_sessions)
    np.save(CACHE_DIR + '/sims_lb_' + str(args.n_sims) + '.npy', itemknn.sims)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmbd', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--t0', type=int, default=56)
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--bw', type=float, default=1.15)
    parser.add_argument('--n_sims', type=int, default=500)
    args = parser.parse_args()
    print(args)

    train_sessions = pd.read_csv(DATA_DIR + '/train_sessions.csv')
    train_purchases = pd.read_csv(DATA_DIR + '/train_purchases.csv')
    train_sessions['ym'] = train_sessions['date'].apply(lambda x: x[:7])
    train_purchases['ym'] = train_purchases['date'].apply(lambda x: x[:7])

    train_sessions.drop(['ym'], axis=1, inplace=True)
    train_purchases.drop(['ym'], axis=1, inplace=True)
    train_sessions['type'] = 'view'
    train_purchases['type'] = 'buy'
    merge = pd.concat([train_sessions, train_purchases]).sort_values(['session_id', 'date'])
    merge = merge.reset_index(drop=True)
    gen_knn_mat(merge)
