
python val.py \
    --max_epoch 100 \
    --mode 'flow' \
    --exp 'D:/data/pose_detection_sample/experiments' \
    --ckpt "D:/data/pose_detection_sample/experiments/steps0000008.pt" \
    --save_every 1000 \
    --dataset 'uavhuman' \
    --root_eval 'D:/data/pose_detection_sample/kinectrgbs_with_missing' \
    --eval_split 'D:/data/pose_detection_sample/train_test_split.json' \
    --num_frames 64 \
    --num_classes 155 \
    --lr 0.002 \
    --batch_size 2 \
    --num_workers 2 \
    --weight_decay 0.0000001
