import time

from data_generator import Data
import joblib
import torch
import argparse
import numpy as np
import glob
import os
from dae_utils.network import AutoEncoder
from config import *


def load_first_stage_model(dump, use_itemknn):
    model = AutoEncoder(**dump['constructor_args'])
    # dump['constructor_args']['item_content']
    dump['network_state_dict']['item_content_embed.weight'] = torch.FloatTensor(
        dump['constructor_args']['item_content'])
    if use_itemknn:
        dump['network_state_dict']['itemknn_embed.weight'] = torch.FloatTensor(dump['constructor_args']['item_knn'])
    model.load_state_dict(dump['network_state_dict'])
    return model


def inference_one_model(model, data, batch_size, device):
    model.eval()
    candidates = torch.from_numpy(data.candidate_val).long().to(device)
    N = data.X_val.shape[0]
    # N = 2000
    idxlist = list(range(N))
    rating_score = []
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        x = data.X_val[idxlist[start_idx:end_idx]]
        x = torch.FloatTensor(x).to(device)
        logit, _ = model.forward({'nums': x})
        logit = logit['nums']
        logit[x > 0] = 0
        logit_candidate = logit[:, candidates]
        rating_score.append(logit_candidate.cpu().numpy())
    rating_score = np.concatenate(rating_score)
    return rating_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='dae_model_knn')  # dae_model_knn, dae_model
    parser.add_argument('--isfinal', type=str, default='lb')
    parser.add_argument('--w1', type=float, default=0.2)
    blend_args = parser.parse_args()
    print(blend_args)

    model_path = MODEL_DIR + "/" + str(blend_args.folder) + '/'
    print('loading models from', model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    topks = (100, 200, 500)
    model_pkls = sorted(glob.glob(os.path.join(model_path, 'model_00?_ft0?.pkl')))
    model_pkls += sorted(glob.glob(os.path.join(model_path, 'model_00?.pkl')))
    print(model_pkls)

    for i, model_pkl in enumerate(model_pkls):
        print('loading', model_pkl)
        dump = joblib.load(model_pkl)
        if 'optimizer' in dump['result']:
            del dump['result']['optimizer']
        args = dump['args']
        if i == 0:
            if blend_args.isfinal == 'final':
                args.mode = 'final'
            data = Data(args)
            model_score = np.zeros((len(data.sess_ids), len(data.candidate_val)))
            if args.use_itemknn:
                knn_feature = data.itemknn_feature
        if args.use_itemknn:
            dump['constructor_args']['item_knn'] = knn_feature
        dump['constructor_args']['cross'] = 0
        dump['constructor_args']['item_content'] = data.item_features
        dump['constructor_args']['decode_feature'] = args.decode_feature
        model = load_first_stage_model(dump, args.use_itemknn)
        model.to(device)
        with torch.no_grad():
            rating_score = inference_one_model(model, data, args.batch_size, device)
        model_score += blend_args.w1 * rating_score

    score_file_name = blend_args.isfinal + "_" + blend_args.folder + "scores.csv"

    start = time.time()
    candidate_ids = np.array([data.item_idx_to_id[x] for x in data.candidate_val])
    with open(score_file_name, 'w') as file:
        file.write('session_id,item_id,score\n')
        for i in range(len(data.sess_ids)):
            sess_id = data.sess_ids[i]
            sess_scores = model_score[i]
            ranks = np.argsort(sess_scores)[::-1]
            item_ids_sorted = candidate_ids[ranks]
            assert all(sess_scores[ranks][np.argsort(item_ids_sorted)] == sess_scores)
            sess_scores_minmax = (sess_scores - min(sess_scores)) / (max(sess_scores) - min(sess_scores))
            for item_id, score in zip(item_ids_sorted, sess_scores_minmax[ranks]):
                file.write(f'{sess_id},{item_id},{score:.4f}\n')

    print(f'write time elapsed {(time.time() - start):.3f}')
