# dae-2stage-inf

## Preparation
1. Download and extract data to a folder in your machine, then specify its path as `DATA_DIR`  in `config.py`: 
```
|-- data
|   |-- candidate_items.csv
|   |-- item_features.csv
|   |-- README.txt
|   |-- REAME_win.txt
|   |-- test_final_sessions.csv
|   |-- test_leaderboard_sessions.csv
|   |-- train_purchases.csv
|   |-- train_sessions.csv
```
2. Create a submission folder in your machine then specify its path as `SUBMISSION_DIR` in `config.py`.

3. Run `preprocessing.py` and `itemknn.py`
```
python preprocessing.py
python itemknn.py
```

## lb scores
```
python inference_dae.py --folder dae_model_knn --isfinal lb
python inference_dae.py --folder dae_model --isfinal lb
python inference_rerank.py --folder dae_model --isfinal lb
python inference_rerank.py --folder dae_model_knn --isfinal lb
python merge_lb.py
```

## final scores
```
python inference_dae.py --folder dae_model_knn --isfinal final
python inference_dae.py --folder dae_model --isfinal final
python inference_rerank.py --folder dae_model --isfinal final
python inference_rerank.py --folder dae_model_knn --isfinal final
python merge_final.py
```

