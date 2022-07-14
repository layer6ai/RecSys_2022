from collections import defaultdict, OrderedDict

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from tqdm import trange

from config import CACHE_DIR, MAX_ITEM_ID, MONTH_LIST
from file_utils import read_pickle, to_pickle


def get_purchase_count(grouped_sessions):
    purchase_count = defaultdict(int)
    for session in grouped_sessions:
        purchase_count[session[-1][0]] += 1
    purchase_count_normalized = {key: value / len(grouped_sessions) for key, value in purchase_count.items()}
    purchase_count = OrderedDict(sorted(purchase_count.items(), key=lambda item: item[1], reverse=True))
    purchase_count_normalized = OrderedDict(sorted(purchase_count_normalized.items(), key=lambda item: item[1], reverse=True))
    return purchase_count, purchase_count_normalized


def get_impression_count(grouped_sessions):
    impression_count = defaultdict(int)
    for session in grouped_sessions:
        item_id_list, _ = zip(*session)
        for item_id in set(item_id_list):
            impression_count[item_id] += 1
    impression_count_normalized = {key: value / len(grouped_sessions) for key, value in impression_count.items()}
    impression_count = OrderedDict(sorted(impression_count.items(), key=lambda item: item[1], reverse=True))
    impression_count_normalized = OrderedDict(sorted(impression_count_normalized.items(), key=lambda item: item[1], reverse=True))
    return impression_count, impression_count_normalized


def get_cooccurrence(grouped_sessions):
    purchase_row, purchase_col, purchase_data = [], [], []
    for session in grouped_sessions:
        item_id_list, _ = zip(*session[: -1])
        target_item_id = session[-1][0]
        for item_id in set(item_id_list):
            purchase_row.append(target_item_id)
            purchase_col.append(item_id)
            purchase_data.append(1)
    purchase_cooccurrence = csr_matrix((purchase_data, (purchase_row, purchase_col)), shape=(MAX_ITEM_ID + 1, MAX_ITEM_ID + 1))
    purchase_cooccurrence_row_normalized = normalize(purchase_cooccurrence, norm='l1', axis=0).astype('float32')
    purchase_cooccurrence_col_normalized = normalize(purchase_cooccurrence, norm='l1', axis=1).astype('float32')

    impression_row, impression_col, impression_data = [], [], []
    for session in grouped_sessions:
        for i in range(len(session)):
            for j in range(i + 1, len(session)):
                impression_row.append(session[i][0])
                impression_col.append(session[j][0])
                impression_data.append(1)
                impression_row.append(session[j][0])
                impression_col.append(session[i][0])
                impression_data.append(1)
    impression_cooccurrence = csr_matrix((impression_data, (impression_row, impression_col)), shape=(MAX_ITEM_ID + 1, MAX_ITEM_ID + 1))
    impression_cooccurrence_row_normalized = normalize(impression_cooccurrence, norm='l1', axis=0).astype('float32')
    impression_cooccurrence_col_normalized = normalize(impression_cooccurrence, norm='l1', axis=1).astype('float32')

    return {
        'purchase_cooccurrence_row_normalized': purchase_cooccurrence_row_normalized,
        'purchase_cooccurrence_col_normalized': purchase_cooccurrence_col_normalized,
        'impression_cooccurrence_row_normalized': impression_cooccurrence_row_normalized,
        'impression_cooccurrence_col_normalized': impression_cooccurrence_col_normalized
    }


def main():
    train_valid_grouped_sessions = [
        read_pickle(path=f'{CACHE_DIR}/train_valid_sessions_{split_index}.pkl')
        for split_index in range(len(MONTH_LIST))
    ]

    metadata_all = OrderedDict()
    for split_index in trange(0, len(MONTH_LIST) + 1):

        #### change from using last month to use all month except this month
        # grouped_sessions = train_valid_grouped_sessions[split_index - 1]
        grouped_sessions = []
        for index in range(len(MONTH_LIST)):
            if index != split_index:
                grouped_sessions.extend(train_valid_grouped_sessions[index])

        metadata_all[split_index] = dict()
        purchase_count_1_month, purchase_count_normalized_1_month = get_purchase_count(grouped_sessions=grouped_sessions)
        metadata_all[split_index]['purchase_count_1_month'] = purchase_count_1_month
        metadata_all[split_index]['purchase_count_normalized_1_month'] = purchase_count_normalized_1_month
        impression_count_1_month, impression_count_normalized_1_month = get_impression_count(grouped_sessions=grouped_sessions)
        metadata_all[split_index]['impression_count_1_month'] = impression_count_1_month
        metadata_all[split_index]['impression_count_normalized_1_month'] = impression_count_normalized_1_month

    #### removed because now we use all months except this month as feature extraction
    # metadata[0] = {key: OrderedDict() for key in metadata[1]}
    # metadata.move_to_end(0, last=False)

    metadata = OrderedDict()
    for split_index in trange(1, len(MONTH_LIST) + 1):
        grouped_sessions = train_valid_grouped_sessions[split_index - 1]
        metadata[split_index] = dict()
        purchase_count_1_month, purchase_count_normalized_1_month = get_purchase_count(grouped_sessions=grouped_sessions)
        metadata[split_index]['purchase_count_1_month'] = purchase_count_1_month
        metadata[split_index]['purchase_count_normalized_1_month'] = purchase_count_normalized_1_month
        impression_count_1_month, impression_count_normalized_1_month = get_impression_count(grouped_sessions=grouped_sessions)
        metadata[split_index]['impression_count_1_month'] = impression_count_1_month
        metadata[split_index]['impression_count_normalized_1_month'] = impression_count_normalized_1_month
    metadata[0] = {key: OrderedDict() for key in metadata[1]}
    metadata.move_to_end(0, last=False)

    metadata_2m = OrderedDict()
    for split_index in trange(1, len(MONTH_LIST) + 1):
        grouped_sessions = train_valid_grouped_sessions[split_index - 1] if split_index == 1 else train_valid_grouped_sessions[split_index - 1] + train_valid_grouped_sessions[split_index - 2]
        metadata_2m[split_index] = dict()
        purchase_count_1_month, purchase_count_normalized_1_month = get_purchase_count(grouped_sessions=grouped_sessions)
        metadata_2m[split_index]['purchase_count_1_month'] = purchase_count_1_month
        metadata_2m[split_index]['purchase_count_normalized_1_month'] = purchase_count_normalized_1_month
        impression_count_1_month, impression_count_normalized_1_month = get_impression_count(grouped_sessions=grouped_sessions)
        metadata_2m[split_index]['impression_count_1_month'] = impression_count_1_month
        metadata_2m[split_index]['impression_count_normalized_1_month'] = impression_count_normalized_1_month
    metadata_2m[0] = {key: OrderedDict() for key in metadata_2m[1]}
    metadata_2m.move_to_end(0, last=False)

    metadata_curm = OrderedDict()
    for split_index in trange(0, len(MONTH_LIST)+1):

        metadata_curm[split_index] = dict()

        # all impression count 
        grouped_sessions = []
        for index in range(len(MONTH_LIST)):
            grouped_sessions.extend(train_valid_grouped_sessions[index])
        impression_count_1_month, impression_count_normalized_1_month = get_impression_count(grouped_sessions=grouped_sessions)

        metadata_curm[split_index]['purchase_count_1_month'] = impression_count_1_month
        metadata_curm[split_index]['purchase_count_normalized_1_month'] = impression_count_normalized_1_month
        
        # current month impression count
        grouped_sessions_curm = train_valid_grouped_sessions[split_index] if split_index < len(MONTH_LIST) else train_valid_grouped_sessions[split_index-1]
        impression_count_1_month, impression_count_normalized_1_month = get_impression_count(grouped_sessions=grouped_sessions_curm)

        metadata_curm[split_index]['impression_count_1_month'] = impression_count_1_month
        metadata_curm[split_index]['impression_count_normalized_1_month'] = impression_count_normalized_1_month

    cooccurrence = OrderedDict()
    for split_index in trange(1, len(MONTH_LIST) + 1):
        cooccurrence[split_index] = get_cooccurrence(grouped_sessions=train_valid_grouped_sessions[split_index - 1])
    cooccurrence[0] = {
        key: csr_matrix((MAX_ITEM_ID + 1, MAX_ITEM_ID + 1), dtype='uint16')
        for key in cooccurrence[1]
    }
    cooccurrence.move_to_end(0, last=False)

    ############################################################################################################################
    # cooccurrence = OrderedDict()
    # for split_index in trange(1, len(MONTH_LIST) + 1):
    #     cooccurrence[split_index] = get_cooccurrence(grouped_sessions=train_valid_grouped_sessions[split_index - 1])

    #     #### change from using last month to use all month except this month
    #     # grouped_sessions = train_valid_grouped_sessions[split_index - 1]
    #     # grouped_sessions = []
    #     # for index in range(len(MONTH_LIST)):
    #     #     if index != split_index:
    #     #         grouped_sessions.extend(train_valid_grouped_sessions[index])
    #     # cooccurrence[split_index] = get_cooccurrence(grouped_sessions=grouped_sessions)

    # #### removed because now we use all months except this month as feature extraction
    # cooccurrence[0] = {
    #     key: csr_matrix((MAX_ITEM_ID + 1, MAX_ITEM_ID + 1), dtype='uint16')
    #     for key in cooccurrence[1]
    # }
    # cooccurrence.move_to_end(0, last=False)
    ############################################################################################################################

    to_pickle(obj=metadata_all, path=f'{CACHE_DIR}/metadata_all.pkl')
    to_pickle(obj=metadata, path=f'{CACHE_DIR}/metadata.pkl')
    to_pickle(obj=metadata_2m, path=f'{CACHE_DIR}/metadata_2m.pkl')
    to_pickle(obj=metadata_curm, path=f'{CACHE_DIR}/metadata_curm.pkl')
    to_pickle(obj=cooccurrence, path=f'{CACHE_DIR}/cooccurrence.pkl')


if __name__ == '__main__':
    main()
