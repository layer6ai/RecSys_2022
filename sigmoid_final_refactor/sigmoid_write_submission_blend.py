import argparse

import numpy as np
import torch
from tqdm import tqdm, trange

from config import CACHE_DIR, SUBMISSION_DIR
from file_utils import read_pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='sigmoid')
    return parser.parse_args()


def main():
    args = parse_args()

    valid_session_ids = read_pickle(path=f'{CACHE_DIR}/test_leaderboard_session_ids.pkl')
    valid_sessions = read_pickle(path=f'{CACHE_DIR}/test_leaderboard_sessions.pkl')
    valid_candidate_items = read_pickle(path=f'{CACHE_DIR}/test_candidate_items.pkl')

    valid_context_indices_list = [sorted(set(item_id for item_id, _ in session)) for session in valid_sessions]
    valid_context_indices_list = [
        [valid_candidate_items.index(item_id) for item_id in session if item_id in valid_candidate_items]
        for session in valid_context_indices_list
    ]
    valid_predictions_mask = np.zeros(shape=(len(valid_sessions), len(valid_candidate_items)), dtype=np.float32)
    for i, context_indices in enumerate(valid_context_indices_list):
        valid_predictions_mask[i, context_indices] = np.NINF

    valid_predictions = valid_predictions_mask
    for filename in tqdm([
        f'{SUBMISSION_DIR}/{args.model_name}_v1_predictions.npy',
        f'{SUBMISSION_DIR}/{args.model_name}_v2_predictions.npy',
        f'{SUBMISSION_DIR}/{args.model_name}_v3_predictions.npy',
        f'{SUBMISSION_DIR}/{args.model_name}_v4_predictions.npy',
        f'{SUBMISSION_DIR}/{args.model_name}_v5_predictions.npy',
    ]):
        temp_predictions = np.load(filename)
        valid_predictions += torch.sigmoid(torch.from_numpy(temp_predictions).cuda()).cpu().numpy()
    valid_predictions = torch.from_numpy(valid_predictions)
    with open(f'{SUBMISSION_DIR}/{args.model_name}_lb_score.csv', 'w') as file_out:
        file_out.write(f'session_id,item_id,score\n')
        for i in trange(len(valid_session_ids)):
            for j in range(len(valid_candidate_items)):
                file_out.write(f'{valid_session_ids[i]},{valid_candidate_items[j]},{valid_predictions[i][j]}\n')


if __name__ == '__main__':
    main()
