CUDA_VISIBLE_DEVICES=0 python experiments/train_agg_sync.py --agg_interval 100 --logs_dir logs &
CUDA_VISIBLE_DEVICES=5 python experiments/train_peer_sync.py --peer_id 0 --agg_interval 100 --batch_size 25 --logs_dir logs &
CUDA_VISIBLE_DEVICES=2 python experiments/train_peer_sync.py --peer_id 1 --agg_interval 100 --batch_size 25 --logs_dir logs &
CUDA_VISIBLE_DEVICES=3 python experiments/train_peer_sync.py --peer_id 2 --agg_interval 100 --batch_size 25 --logs_dir logs &
CUDA_VISIBLE_DEVICES=4 python experiments/train_peer_sync.py --peer_id 3 --agg_interval 100 --batch_size 25 --logs_dir logs

