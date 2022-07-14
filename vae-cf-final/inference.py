from preprocessing import preprocessing
import os
import torch
from data_generator import Data
from multivae import MultiVAE
import numpy as np

def generate_lb_final(ratings, sess_ids, item_idx_to_id, candidates, len_lb, lb_file_name, final_file_name):

    blend_ratings = torch.stack(ratings)
    blend_ratings = torch.mean(blend_ratings, dim=0).numpy()

    candidates_id = []
    for i in candidates:
        candidates_id.append(item_idx_to_id[i])
    candidates_id = np.tile(candidates_id, len_lb)

    lb_ratings = blend_ratings[:len_lb, :].flatten()
    lb_sess = np.repeat(sess_ids[:len_lb], len(candidates))

    output = np.stack((lb_sess, candidates_id, lb_ratings), axis=1)
    np.savetxt(lb_file_name, output, delimiter=',', fmt=['%d','%d','%.5f'], header='session_id,item_id,score', comments='')

    final_ratings = blend_ratings[len_lb:, :].flatten()
    final_sess = np.repeat(sess_ids[len_lb:], len(candidates))

    output = np.stack((final_sess, candidates_id, final_ratings), axis=1)
    np.savetxt(final_file_name, output, delimiter=',', fmt=['%d','%d','%.5f'], header='session_id,item_id,score', comments='')

if __name__ == "__main__":

    # preproces data
    if not os.path.exists('./data/item_features.pkl'):
        preprocessing()

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    data = Data()

    # version 1
    files = ['./model/zhaolin_v1_1.pt', './model/zhaolin_v1_2.pt', './model/zhaolin_v1_3.pt']
    ratings = []
    for file in files:
        model = MultiVAE(data, device, q_dims=[600, 200], p_dims=[200])
        model.load_state_dict(torch.load(file))
        model.to(device)
        model.eval()
        ratings.append(model.compute_rating())

    generate_lb_final(ratings, data.sess_ids, data.item_idx_to_id, data.candidate_val, data.len_lb, \
                        'zhaolin_lb_score_v1.csv', 'zhaolin_final_score_v1.csv')

    # version 2
    files = ['./model/zhaolin_v2_1.pt', './model/zhaolin_v2_2.pt', './model/zhaolin_v2_3.pt']
    ratings = []
    for file in files:
        model = MultiVAE(data, device, q_dims=[600, 300], p_dims=[300])
        model.load_state_dict(torch.load(file))
        model.to(device)
        model.eval()
        ratings.append(model.compute_rating())

    generate_lb_final(ratings, data.sess_ids, data.item_idx_to_id, data.candidate_val, data.len_lb, \
                        'zhaolin_lb_score_v2.csv', 'zhaolin_final_score_v2.csv')