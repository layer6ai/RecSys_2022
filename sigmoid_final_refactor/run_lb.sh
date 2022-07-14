## cache dataset
python cache_dataset.py
python cache_features_all_months.py
python cache_test_dataset.py

## inference 
CUDA_VISIBLE_DEVICES=0 python sigmoid_submit_new_feature.py --model_name sigmoid_v1 &
CUDA_VISIBLE_DEVICES=1 python sigmoid_submit_new_feature.py --model_name sigmoid_v2 &
CUDA_VISIBLE_DEVICES=2 python sigmoid_submit_new_feature.py --model_name sigmoid_v3 &
CUDA_VISIBLE_DEVICES=3 python sigmoid_submit_new_feature.py --model_name sigmoid_v4 &
CUDA_VISIBLE_DEVICES=4 python sigmoid_submit_new_feature.py --model_name sigmoid_v5 &
CUDA_VISIBLE_DEVICES=5 python sigmoid_submit_new_feature.py --model_name sigmoid_v6 &
CUDA_VISIBLE_DEVICES=6 python sigmoid_submit_new_feature.py --model_name sigmoid_v7 &
CUDA_VISIBLE_DEVICES=7 python sigmoid_submit_new_feature.py --model_name sigmoid_v8

## generate blend scores
python sigmoid_write_submission_blend.py --model_name sigmoid

