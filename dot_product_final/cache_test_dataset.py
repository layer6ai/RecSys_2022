import pandas as pd
from tqdm import tqdm

from file_utils import to_pickle


def main():
    test_leaderboard_sessions = pd.read_csv('data/test_leaderboard_sessions.csv', parse_dates=['date'])
    test_leaderboard_sessions = test_leaderboard_sessions.sort_values(by=['session_id', 'date'], ascending=True)
    test_leaderboard_session_ids = sorted(test_leaderboard_sessions['session_id'].unique())
    test_leaderboard_grouped_sessions = []
    for session_id in tqdm(test_leaderboard_session_ids):
        session_df = test_leaderboard_sessions.query(f'session_id == {session_id}')
        test_leaderboard_grouped_sessions.append(session_df[['item_id', 'date']].values.tolist())
    to_pickle(obj=test_leaderboard_session_ids, path='cache/test_leaderboard_session_ids.pkl')
    to_pickle(obj=test_leaderboard_grouped_sessions, path='cache/test_leaderboard_sessions.pkl')

    test_final_sessions = pd.read_csv('data/test_final_sessions.csv', parse_dates=['date'])
    test_final_sessions = test_final_sessions.sort_values(by=['session_id', 'date'], ascending=True)
    test_final_session_ids = sorted(test_final_sessions['session_id'].unique())
    test_final_grouped_sessions = []
    for session_id in tqdm(test_final_session_ids):
        session_df = test_final_sessions.query(f'session_id == {session_id}')
        test_final_grouped_sessions.append(session_df[['item_id', 'date']].values.tolist())
    to_pickle(obj=test_final_session_ids, path='cache/test_final_session_ids.pkl')
    to_pickle(obj=test_final_grouped_sessions, path='cache/test_final_sessions.pkl')

    test_candidate_items = pd.read_csv('data/candidate_items.csv')
    test_candidate_items = sorted(test_candidate_items['item_id'].unique())
    to_pickle(obj=test_candidate_items, path='cache/test_candidate_items.pkl')


if __name__ == '__main__':
    main()
