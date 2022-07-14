import datetime
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import NUM_FEATURE_CATEGORY_ID, MONTH_LIST
from file_utils import to_pickle


def main():
    start = time.time()

    train_valid_sessions = pd.read_csv('data/train_sessions.csv', parse_dates=['date'])

    train_valid_purchases = pd.read_csv('data/train_purchases.csv', parse_dates=['date'])

    assert train_valid_sessions['session_id'].nunique() == train_valid_purchases['session_id'].nunique()

    train_valid_sessions = pd.concat([train_valid_sessions, train_valid_purchases], axis=0)
    train_valid_sessions = train_valid_sessions.sort_values(by=['session_id', 'date'], ascending=True)

    train_valid_sessions['month'] = train_valid_sessions['date'].apply(lambda date: f'{date.year}-{date.month:02}')
    assert sorted(train_valid_sessions['month'].unique()) == MONTH_LIST

    split_date = datetime.datetime(year=2021, month=5, day=1, hour=0, minute=0, second=0)

    train_sessions = train_valid_sessions[train_valid_sessions['date'] < split_date]
    valid_sessions = train_valid_sessions[train_valid_sessions['date'] >= split_date]

    train_session_ids = train_sessions['session_id'].unique()
    valid_session_ids = valid_sessions['session_id'].unique()

    assert len(np.intersect1d(ar1=train_session_ids, ar2=valid_session_ids, assume_unique=True)) == 0

    train_grouped_sessions = []
    for split_index, month in enumerate(tqdm(MONTH_LIST[: -1])):
        split_sessions = train_sessions[train_sessions['month'] == month]
        split_sessions = split_sessions.sort_values(by=['session_id', 'date'], ascending=True)
        split_sessions = split_sessions.reset_index(drop=True)
        assert len(split_sessions) > 0

        grouped_sessions = split_sessions.groupby('session_id').apply(
            lambda x: [[row['item_id'], row['date']] for _, row in x.iterrows()]
        ).values.tolist()
        train_grouped_sessions.extend(grouped_sessions)

        impression_items = sorted(set(item_id for session in grouped_sessions for item_id, _ in session))
        candidate_items = sorted(set(session[-1][0] for session in grouped_sessions))

        to_pickle(obj=grouped_sessions, path=f'cache/train_sessions_{split_index}.pkl')
        to_pickle(obj=impression_items, path=f'cache/train_impression_items_{split_index}.pkl')
        to_pickle(obj=candidate_items, path=f'cache/train_candidate_items_{split_index}.pkl')
        to_pickle(obj=grouped_sessions, path=f'cache/train_valid_sessions_{split_index}.pkl')
        to_pickle(obj=impression_items, path=f'cache/train_valid_impression_items_{split_index}.pkl')
        to_pickle(obj=candidate_items, path=f'cache/train_valid_candidate_items_{split_index}.pkl')

    assert len(train_session_ids) == len(train_grouped_sessions)
    to_pickle(obj=train_grouped_sessions, path='cache/train_sessions.pkl')

    valid_sessions = valid_sessions.sort_values(by=['session_id', 'date'], ascending=True)
    valid_sessions = valid_sessions.reset_index(drop=True)

    valid_grouped_sessions = valid_sessions.groupby('session_id').apply(
        lambda x: [[row['item_id'], row['date']] for _, row in x.iterrows()]
    ).values.tolist()

    impression_items = sorted(set(item_id for session in valid_grouped_sessions for item_id, _ in session))
    candidate_items = sorted(set(session[-1][0] for session in valid_grouped_sessions))

    assert len(valid_session_ids) == len(valid_grouped_sessions)
    to_pickle(obj=valid_grouped_sessions, path='cache/valid_sessions.pkl')
    to_pickle(obj=impression_items, path='cache/valid_impression_items.pkl')
    to_pickle(obj=candidate_items, path='cache/valid_candidate_items.pkl')
    to_pickle(obj=valid_grouped_sessions, path=f'cache/train_valid_sessions_{len(MONTH_LIST) - 1}.pkl')
    to_pickle(obj=impression_items, path=f'cache/train_valid_impression_items_{len(MONTH_LIST) - 1}.pkl')
    to_pickle(obj=candidate_items, path=f'cache/train_valid_candidate_items_{len(MONTH_LIST) - 1}.pkl')

    train_valid_grouped_sessions = train_grouped_sessions + valid_grouped_sessions
    to_pickle(obj=train_valid_grouped_sessions, path='cache/train_valid_sessions.pkl')

    assert len(train_valid_grouped_sessions) == train_valid_sessions['session_id'].nunique()
    assert sum(len(session) for session in train_valid_grouped_sessions) == len(train_valid_sessions)
    assert all(session == sorted(session, key=lambda x: x[1], reverse=False) for session in train_valid_grouped_sessions)

    train_item_features = pd.read_csv('data/item_features.csv')
    train_item_counts = train_sessions['item_id'].value_counts().to_dict()
    train_item_features['item_count'] = train_item_features['item_id'].apply(
        lambda item_id: train_item_counts.get(item_id, 0)
    )
    train_feature_counts = train_item_features.groupby(['feature_category_id', 'feature_value_id'])['item_count'].sum()
    train_feature_counts = OrderedDict([
        (feature_category_id, OrderedDict([
            (
                row['feature_value_id'],
                row['item_count']
            ) for _, row in train_feature_counts[feature_category_id].to_frame().reset_index().sort_values(
                by=['item_count', 'feature_value_id'],
                ascending=[False, True]
            ).iterrows()
        ])) for feature_category_id in range(1, NUM_FEATURE_CATEGORY_ID + 1)
    ])
    to_pickle(obj=train_feature_counts, path='cache/train_feature_counts.pkl')

    train_item_counts = OrderedDict(sorted(train_item_counts.items(), key=lambda item: item[1], reverse=True))
    to_pickle(obj=train_item_counts, path='cache/train_item_counts.pkl')

    train_valid_item_features = pd.read_csv('data/item_features.csv')
    train_valid_item_counts = train_valid_sessions['item_id'].value_counts().to_dict()
    train_valid_item_features['item_count'] = train_valid_item_features['item_id'].apply(
        lambda item_id: train_valid_item_counts.get(item_id, 0)
    )
    train_valid_feature_counts = train_valid_item_features.groupby(['feature_category_id', 'feature_value_id'])['item_count'].sum()
    train_valid_feature_counts = OrderedDict([
        (feature_category_id, OrderedDict([
            (
                row['feature_value_id'],
                row['item_count']
            ) for _, row in train_valid_feature_counts[feature_category_id].to_frame().reset_index().sort_values(
                by=['item_count', 'feature_value_id'],
                ascending=[False, True]
            ).iterrows()
        ])) for feature_category_id in range(1, NUM_FEATURE_CATEGORY_ID + 1)
    ])
    to_pickle(obj=train_valid_feature_counts, path='cache/train_valid_feature_counts.pkl')

    train_valid_item_counts = OrderedDict(sorted(train_valid_item_counts.items(), key=lambda item: item[1], reverse=True))
    to_pickle(obj=train_valid_item_counts, path='cache/train_valid_item_counts.pkl')

    item_features = pd.read_csv('data/item_features.csv')
    item_features = {
        item_id: item_features.query(
            f'item_id == {item_id}'
        )[['feature_category_id', 'feature_value_id']].groupby('feature_category_id')['feature_value_id'].apply(list).to_dict()
        for item_id in tqdm(sorted(item_features['item_id'].unique()))
    }
    to_pickle(obj=item_features, path='cache/item_features.pkl')

    end = time.time()

    print(f'Time elapsed: {end - start:.2f} seconds...')


if __name__ == '__main__':
    main()
