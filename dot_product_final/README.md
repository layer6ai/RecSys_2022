# dot_product_final

## Environment
```
pytorch >= 1.8.1
pandas == 1.4.1
tqdm
```

## Data Preparation

After cloning this repo, download and extract data to the `data` folder:

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

## Submission

```
python cache_dataset.py
python cache_test_dataset.py
python dot_product_inference.py --split leaderboard
python dot_product_inference.py --split final
```
Running the above commands creates `leaderboard_raw_scores.csv` and `final_raw_scores.csv` under the `output` folder, which can then be used for blending.
