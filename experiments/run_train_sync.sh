CUDA_VISIBLE_DEVICES=0 python train_agg_sync.py --agg_interval 100 
CUDA_VISIBLE_DEVICES=1 python train_agg_sync.py --peer_id 0 --agg_interval 100 --batch_size 25
CUDA_VISIBLE_DEVICES=2 python train_agg_sync.py --peer_id 1 --agg_interval 100 --batch_size 25
CUDA_VISIBLE_DEVICES=3 python train_agg_sync.py --peer_id 2 --agg_interval 100 --batch_size 25
CUDA_VISIBLE_DEVICES=4 python train_agg_sync.py --peer_id 3 --agg_interval 100 --batch_size 25

