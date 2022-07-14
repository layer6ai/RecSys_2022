import os
from inference_dae import load_first_stage_model
from data_generator import Data
import joblib
import numpy as np
import torch
import argparse
import glob
import time
from torch import nn
from dae_utils.network import Transformer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from config import *
from dae_utils.torch_data import SecondStageTest, SecondStageFirst

_ae_body_options = {
    'transformer': Transformer
}


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


class MinMaxScaler3D(MinMaxScaler):
    def fit_transform(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0] * X.shape[1], X.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=X.shape)


def featurize_and_inference_mul(non_sampled_knn, models, dataloader, candidate_idx, device, num_k):
    rating_list, session_feature, last_item_content, rating_scores = [], [], [], []
    score_scaler = MinMaxScaler3D()

    for i, batch_x_in in enumerate(dataloader):
        batch_x_in = batch_x_in.to(device)
        logit_candidates, last_hiddens = [], []
        logit_all = []
        for model in models:
            model.eval()
            logit, last_hidden = model.forward({'nums': batch_x_in}, featurize=True)
            logit = logit['nums']
            logit[batch_x_in > 0] = 0
            logit_all.append(logit.detach().cpu())
            logit_candidates.append(logit[:, candidate_idx])
            last_hiddens.append(last_hidden.detach().cpu())
        logit_candidate = torch.mean(torch.stack(logit_candidates), dim=0)
        last_hidden = torch.cat(last_hiddens, dim=1)
        rating_score, rating_K = torch.topk(logit_candidate, k=num_k)
        rating_list.append(rating_K.cpu().numpy())
        last_item = torch.argmax(batch_x_in, dim=1)
        topk_idx = np.array(candidate_idx)[rating_K.cpu().numpy()]
        last_item_knn_to_all = non_sampled_knn[last_item.detach().cpu().numpy()]
        last_item_knn_to_topk = last_item_knn_to_all[np.indices(topk_idx.shape)[0], topk_idx]
        last_item_knn_to_topk = np.expand_dims(last_item_knn_to_topk, -1)
        each_model_score = torch.stack([torch.gather(x, 1, rating_K) for x in logit_candidates], dim=-1).cpu().numpy()
        scores = score_scaler.fit_transform(np.concatenate((last_item_knn_to_topk, each_model_score), axis=-1))
        rating_scores.append(scores)
        last_item_content.append(model.item_content_embed(last_item).detach().cpu().numpy())
        session_feature.append(last_hidden.detach().cpu().numpy())
    rating_list = np.array(candidate_idx)[np.concatenate(rating_list)]
    session_feature = np.concatenate(session_feature)
    last_item_content = np.concatenate(last_item_content)
    rating_scores = np.concatenate(rating_scores)
    all_session_feature = np.concatenate((session_feature, last_item_content), axis=1)
    return rating_list, rating_scores, all_session_feature


def prepare_eval(first_stage_models, data, non_sampled_knn, batch_size, device, num_k, knn_features):
    test_dataloader = DataLoader(SecondStageFirst(data.X_val),
                                 batch_size=batch_size, num_workers=5,
                                 pin_memory=True, shuffle=False)
    with torch.no_grad():
        first_stage_predictions, first_stage_scores, session_feature = featurize_and_inference_mul(non_sampled_knn,
                                                                                                   first_stage_models,
                                                                                                   test_dataloader,
                                                                                                   data.candidate_val,
                                                                                                   device, num_k)
    second_stage_val_dl = DataLoader(
        SecondStageTest(data.item_features, session_feature, first_stage_predictions, first_stage_scores,
                        itemknn=knn_features),
        batch_size, num_workers=5, pin_memory=True, shuffle=False)
    return second_stage_val_dl


class SecondStageModel(nn.Module):
    def __init__(self, input_dim, num_k, body_network='deepstack', body_network_cfg=dict()):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.body_network = body_network
        body_network = _ae_body_options[body_network]
        self.body_network_cfg = body_network_cfg
        self.input_dim = input_dim
        self.sess_body = body_network(in_features=input_dim[0], **body_network_cfg)
        self.cand_body = body_network(in_features=input_dim[1], **body_network_cfg)
        self.output_layer = nn.Linear(self.cand_body.output_shape * 2, 1)
        self.num_k = num_k
        print(self.input_dim)

    def softmax_loss(self, predict, target):
        predict = predict.view(rerank_args.train_batch_size, -1)
        target = target.view(rerank_args.train_batch_size, -1)
        return -torch.sum(torch.sum(torch.nn.functional.log_softmax(predict, 1) * target, -1))

    def forward(self, sess_feature, cand_feature):
        # normalize each tower's input
        sess_feature = torch.nn.functional.normalize(sess_feature)
        cand_feature = torch.nn.functional.normalize(cand_feature)

        last_hidden_sess = self.sess_body(sess_feature)
        last_hidden_sess = last_hidden_sess.unsqueeze(1).repeat(1, self.num_k, 1)
        last_hidden_sess = last_hidden_sess.view(-1, last_hidden_sess.shape[-1])

        last_hidden_cand = self.cand_body(cand_feature)
        logit = self.output_layer(torch.cat((last_hidden_cand, last_hidden_sess), dim=1))
        return logit

    def inference(self, dl, max_k, device):
        rating_list = []
        rating_scores = []
        start = time.time()
        for i, (batch_session_feature, batch_cand_feature, batch_first_stage_lists) in enumerate(dl):
            batch_session_feature = batch_session_feature.to(device)
            batch_cand_feature = batch_cand_feature.to(device)
            batch_cand_feature = batch_cand_feature.view(-1, batch_cand_feature.shape[-1])
            batch_first_stage_lists = batch_first_stage_lists.to(device)
            logits = self.forward(batch_session_feature, batch_cand_feature)
            logits = logits.view(batch_first_stage_lists.shape[0], -1)
            rating_score, rating_k = torch.topk(logits, max_k)
            rating_list.append(torch.gather(batch_first_stage_lists, 1, rating_k).cpu().numpy())
            rating_scores.append(torch.gather(logits, 1, rating_k).detach().cpu().numpy())
            print('\rinference - batch {:3d} / {:3d} - time {:0.2f}s'.format(i + 1, len(dl), time.time() - start),
                  end=' ')
        rating_list = np.concatenate(rating_list)
        rating_scores = np.concatenate(rating_scores)
        return rating_list, rating_scores

    def save(self, path_to_model_dump, rerank_args, result=None):
        model_state_dict = dict(
            constructor_args=dict(
                input_dim=self.input_dim,
                num_k=self.num_k,
                body_network=self.body_network,
                body_network_cfg=self.body_network_cfg,
            ),
            result=result,
            rerank_args=rerank_args,
            network_state_dict={k: v.cpu() for k, v in self.state_dict().items()}
        )
        joblib.dump(model_state_dict, path_to_model_dump)


def load_second_stage_model(dump):
    model = SecondStageModel(**dump['constructor_args'])
    model.load_state_dict(dump['network_state_dict'])
    return model.to(device)


def blend_scores(topk_list, topk_scores, sess_ids, item_idx_to_id, name=""):
    for i in range(len(topk_list)):
        for j in range(len(topk_list[i])):
            topk_list[i, j] = item_idx_to_id[topk_list[i, j]]
    topk_list = topk_list.flatten()
    topk_scores = topk_scores.flatten()
    session_ids = np.repeat(sess_ids, 100)
    output = np.stack((session_ids, topk_list, topk_scores), axis=1)
    print('model scores written to ' + name + '.csv')
    np.savetxt("./scores/" + name + '.csv', output, delimiter=',', fmt=['%d', '%d', '%f'],
               header='session_id,item_id,score', comments='')


def generate_submission_second(dl, model):
    model = model.eval()
    with torch.no_grad():
        predictions, scores = model.inference(dl, rerank_args.num_k, device)
    return predictions, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='dae_model')  # 04301442 05022051
    parser.add_argument('--isfinal', type=str, default='final')
    parser.add_argument('--cut_train', type=float, default=0.4)
    parser.add_argument('--f_batch_size', type=int, default=200)
    parser.add_argument('--num_k', type=int, default=150)
    parser.add_argument('--knn_rate', type=float, default=0.03)
    rerank_args = parser.parse_args()
    print(rerank_args)

    SEED = 1234
    set_seed(SEED)
    print("SEED =", SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    first_stage_path = MODEL_DIR + '/' + rerank_args.folder
    model_pkls = sorted(glob.glob(os.path.join(first_stage_path, 'model_00*.pkl')))
    print(first_stage_path, model_pkls)

    first_stage_models = []
    for i, model_pkl in enumerate(model_pkls):
        dump = joblib.load(model_pkl)
        args = dump['args']
        if i == 0:
            args.sess_feature = 0
            args.cut_train = rerank_args.cut_train
            if rerank_args.isfinal == 'final':
                args.mode = 'final'
            data = Data(args)
            if args.use_itemknn:
                knn_feature = data.itemknn_feature
        if args.use_itemknn:
            dump['constructor_args']['item_knn'] = knn_feature
        dump['constructor_args']['item_content'] = data.item_features
        print('loading first stage model checkpoint:', model_pkl)
        first_stage_model = load_first_stage_model(dump, args.use_itemknn)
        first_stage_model.to(device)
        first_stage_models.append(first_stage_model)

    non_sampled_knn = data.process_knn_scores(0)
    knn_features = data.process_knn_scores(rerank_args.knn_rate) if rerank_args.knn_rate else None

    second_stage_model_pkl = sorted(
        glob.glob(os.path.join(MODEL_DIR + '/' + 'rerank_' + rerank_args.folder, 'model_iind_0?.pkl')))
    print(second_stage_model_pkl)
    dump = joblib.load(second_stage_model_pkl[0])
    second_stage_model = load_second_stage_model(dump).to(device)
    second_stage_val_dl = prepare_eval(first_stage_models, data, non_sampled_knn, rerank_args.f_batch_size,
                                       device, rerank_args.num_k, knn_features)

    score_file_name = rerank_args.isfinal + "_" + rerank_args.folder + "_iind.csv"
    rating_lists, rating_scores = generate_submission_second(second_stage_val_dl, second_stage_model)

    start = time.time()
    with open(score_file_name, 'w') as file:
        file.write('session_id,item_id,score\n')
        for i in range(len(data.sess_ids)):
            sess_id = data.sess_ids[i]
            sess_scores = rating_scores[i]
            item_ids_sorted = [data.item_idx_to_id[x] for x in rating_lists[i]]
            sess_scores_minmax = (sess_scores - min(sess_scores)) / (max(sess_scores) - min(sess_scores))
            for item_id, score in zip(item_ids_sorted, sess_scores_minmax):
                file.write(f'{sess_id},{item_id},{score:.4f}\n')
