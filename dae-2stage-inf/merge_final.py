import pandas as pd

print('loading dae model scores...')
model_1 = pd.read_csv('./final_dae_model_knnscores.csv')
model_1.rename(columns={'score': 'score1'}, inplace=True)
model_2 = pd.read_csv('./final_dae_modelscores.csv')
model_2.rename(columns={'score': 'score2'}, inplace=True)

model_first = model_1.merge(model_2, on=['session_id', 'item_id'])
model_first['score_first_stage'] = model_first['score1'] + model_first['score2']
model_first.drop('score1', axis=1, inplace=True)
model_first.drop('score2', axis=1, inplace=True)

print('loading rerank model scores...')
model_1_second_stage = pd.read_csv('./final_dae_model_knn_iind.csv')
model_2_second_stage = pd.read_csv('./final_dae_model_iind.csv')
model_1_second_stage.rename(columns={'score': 'score1_2'}, inplace=True)
model_2_second_stage.rename(columns={'score': 'score2_2'}, inplace=True)

model_second_stage = model_1_second_stage.merge(model_2_second_stage, on=['session_id', 'item_id'], how='outer')
model_second_stage = model_second_stage.fillna(0)
model_second_stage['score_second_total'] = model_second_stage['score1_2'] + model_second_stage['score2_2']

print('merging model scores...')
model_merge = model_first[['session_id', 'item_id', 'score_first_stage']].merge(
    model_second_stage[['session_id', 'item_id', 'score_second_total']],
    on=['session_id', "item_id"], how='left')
model_merge = model_merge.fillna(0)
model_merge['score_final'] = model_merge['score_first_stage'] + model_merge['score_second_total']

print('saving merged scores...')
score_csv = model_merge[['session_id', 'item_id', 'score_final']].copy()
score_csv['score_final'] = score_csv['score_final'].round(4)
score_csv.rename(columns={"score_final": 'score'}, inplace=True)
score_csv.to_csv('./jianing_final_scores.csv', header=True, index=False)
